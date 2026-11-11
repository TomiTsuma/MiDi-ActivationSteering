"""Phase One activation capture against the frozen MiDi checkpoint.

Three modes, all sharing one setup:

  mode=encode     (default) the Phase One capture. Real QM9 molecules with known
                  DFT labels are noised to a chosen timestep, passed through the
                  frozen denoiser, and the pooled activations are stored.
  mode=liveness   the site liveness screen. Injects a random direction at each
                  (layer, channel) and reports whether the denoiser's OUTPUT
                  changes at all. A site whose output is bit-identical is
                  causally dead and no steering vector can ever work there.
  mode=rotation   the Section 3.6 validation: pass each molecule twice, once
                  with rotated input coordinates, and require the scalar
                  channels to agree.

Why encode and not generate
---------------------------
Five of the seven target properties (polarisability, heat capacity, dipole
moment, HOMO, LUMO) are DFT labels that exist only for real QM9 molecules.
A molecule the model invents during generation has no such label, and obtaining
one would mean running DFT on every sample. So RQ1, "how much of the variance
in a target property is linearly decodable from the activations", is answerable
only on labelled molecules passed forward through the denoiser. Generation-time
capture is what Phase Two needs; it is not what Phase One can probe.

Store layout
------------
    <store>/pooled_L00.npy ... pooled_L11.npy   each (T_cap, N, 1408) float32
    <store>/labels.npz                          DFT labels, dataset-aligned
    <store>/manifest.json                       everything needed to reproduce

Feature axis (1408), see FEATURE_SLICES:
    [   0: 128]  y, the global graph vector
    [ 128:1152]  X pooled 4 ways (mean, min, max, var) x 256
    [1152:1408]  E pooled 4 ways (mean, min, max, var) x 64

The four statistics replicate Xtoy and Etoy in midi/models/layers.py, which is
how the model itself builds y. Note that both use the VARIANCE, not the standard
deviation: there is no sqrt in layers.py. Mean-pooling alone would discard three
of the four statistics the model uses.

Layer 11 caveat
---------------
The y block is captured at every layer including 11, but the 12th layer's y
reaches no output head (mlp_out_y is commented out in transformer_model.py, and
y = y_to_out passes the INPUT y through). It is therefore probe-viable and
intervention-dead. It is kept deliberately, as a negative control for the
liveness screen; it must never be used as an injection site.
"""

import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

import midi._compat  # noqa: F401  -- torch.load default; must precede any load
import hydra
import omegaconf

from midi import rng, utils
from midi import qm9_labels
from midi.datasets import qm9_dataset
from midi.diffusion_model import FullDenoisingDiffusion

N_MAX = 29              # QM9 with explicit hydrogens. Global, never the batch max.
D_Y, D_X, D_E = 128, 256, 64
F_TOTAL = D_Y + 4 * D_X + 4 * D_E       # 1408

FEATURE_SLICES = {
    "y":      (0, D_Y),
    "X_mean": (D_Y + 0 * D_X, D_Y + 1 * D_X),
    "X_min":  (D_Y + 1 * D_X, D_Y + 2 * D_X),
    "X_max":  (D_Y + 2 * D_X, D_Y + 3 * D_X),
    "X_var":  (D_Y + 3 * D_X, D_Y + 4 * D_X),
    "E_mean": (D_Y + 4 * D_X + 0 * D_E, D_Y + 4 * D_X + 1 * D_E),
    "E_min":  (D_Y + 4 * D_X + 1 * D_E, D_Y + 4 * D_X + 2 * D_E),
    "E_max":  (D_Y + 4 * D_X + 2 * D_E, D_Y + 4 * D_X + 3 * D_E),
    "E_var":  (D_Y + 4 * D_X + 3 * D_E, D_Y + 4 * D_X + 4 * D_E),
}
DEAD_Y_LAYER = 11       # y at this layer's output reaches no output head


# ---------------------------------------------------------------------------
# Pooling, replicating midi/models/layers.py Xtoy and Etoy
# ---------------------------------------------------------------------------

def pool_nodes(X, node_mask):
    """(bs, n, dx) -> (bs, 4*dx). Mirrors Xtoy.forward minus its Linear."""
    x_mask = node_mask.unsqueeze(-1).expand(-1, -1, X.shape[-1]).to(X.dtype)
    denom = x_mask.sum(dim=1).clamp_min(1.0)
    float_imask = 1.0 - x_mask
    m = (X * x_mask).sum(dim=1) / denom
    mi = (X + 1e5 * float_imask).min(dim=1)[0]
    ma = (X - 1e5 * float_imask).max(dim=1)[0]
    var = (((X - m[:, None, :]) ** 2) * x_mask).sum(dim=1) / denom   # variance, as in layers.py
    return torch.cat((m, mi, ma, var), dim=-1)


def pool_edges(E, node_mask):
    """(bs, n, n, de) -> (bs, 4*de). Mirrors Etoy.forward minus its Linear."""
    x_mask = node_mask.unsqueeze(-1)
    e_mask = (x_mask.unsqueeze(2) * x_mask.unsqueeze(1)).expand(-1, -1, -1, E.shape[-1]).to(E.dtype)
    denom = e_mask.sum(dim=(1, 2)).clamp_min(1.0)
    float_imask = 1.0 - e_mask
    m = (E * e_mask).sum(dim=(1, 2)) / denom
    mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
    ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
    var = (((E - m[:, None, None, :]) ** 2) * e_mask).sum(dim=(1, 2)) / denom
    return torch.cat((m, mi, ma, var), dim=-1)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class CaptureCtx:
    """Mutable context the hooks read. Set `enabled` and `buf` per forward."""

    def __init__(self, n_layers):
        self.enabled = False
        self.buf = [None] * n_layers
        self.n_layers = n_layers

    def reset(self):
        self.buf = [None] * self.n_layers


def make_capture_hook(ctx, layer_idx):
    def hook(module, inputs, output):
        if not ctx.enabled:
            return None
        nm = output.node_mask
        feats = torch.cat((output.y,
                           pool_nodes(output.X, nm),
                           pool_edges(output.E, nm)), dim=-1)
        ctx.buf[layer_idx] = feats.detach().to(torch.float32).cpu()
        return None
    return hook


def make_inject_hook(state):
    """Adds state['alpha'] * state['v'] to state['channel'] of the layer output.

    Used by mode=liveness only. Writing to 'E' preserves symmetry by
    construction, and writing to 'pos' is left out: PlaceHolder.mask()
    re-centres coordinates at the end of every layer, so a constant direction
    added there is annihilated exactly (verified: residual 1.1e-16).
    """
    def hook(module, inputs, output):
        ch, v, a = state["channel"], state["v"], state["alpha"]
        if ch is None:
            return None
        if ch == "y":
            output.y = output.y + a * v
        elif ch == "X":
            output.X = output.X + a * v.view(1, 1, -1) * output.node_mask.unsqueeze(-1)
        elif ch == "E":
            w = a * v.view(1, 1, 1, -1)
            output.E = output.E + 0.5 * (w + w)      # symmetric by construction
            output.E = 0.5 * (output.E + output.E.transpose(1, 2))
        else:
            raise ValueError(f"unknown channel {ch}")
        return output
    return hook


# ---------------------------------------------------------------------------
# Deterministic noising to a chosen timestep
# ---------------------------------------------------------------------------

def pad_dense(dense, n_max):
    """Pad a PlaceHolder from the batch max to a fixed n_max.

    Fixed padding matters for reproducibility: the per-molecule uniform budget
    is 2*n_max + n_max**2, so a batch-dependent n_max would give the same
    molecule different randomness in a different batch.
    """
    bs, n = dense.node_mask.shape
    if n == n_max:
        return dense
    if n > n_max:
        raise ValueError(f"batch has {n} atoms, exceeds n_max={n_max}")
    p = n_max - n
    dense.X = F.pad(dense.X, (0, 0, 0, p))
    dense.charges = F.pad(dense.charges, (0, 0, 0, p))
    dense.pos = F.pad(dense.pos, (0, 0, 0, p))
    dense.E = F.pad(dense.E, (0, 0, 0, p, 0, p))
    dense.node_mask = F.pad(dense.node_mask, (0, p), value=False)
    return dense.mask()


def noise_to_t(nm, dense, t_int_scalar, unif, pos_noise):
    """Replicate NoiseModel.apply_noise at a CHOSEN timestep with GIVEN randomness.

    apply_noise draws t_int ~ Uniform{1..T} per molecule and reads the global RNG
    stream for both the categorical and the position noise. For capture we need a
    named timestep and per-molecule reproducible randomness, so this is a
    tape-driven twin. The algebra is identical.

        t_float = t_int / T
        Qtb     = cumulative transition matrix to t
        probX   = X0 @ Qtb.X                     (and likewise charges, E)
        pos_t   = alpha_bar(t) * pos0 + sigma_bar(t) * eps,  eps mean-free
    """
    bs = dense.X.size(0)
    dev = dense.X.device
    t_int = torch.full((bs, 1), int(t_int_scalar), dtype=torch.long, device=dev)
    t_float = t_int.float() / nm.T

    Qtb = nm.get_Qt_bar(t_int=t_int)
    probX = dense.X.float() @ Qtb.X
    prob_charges = dense.charges.float() @ Qtb.charges
    probE = dense.E.float() @ Qtb.E.unsqueeze(1)

    Xi, Ci, Ei = rng.sample_discrete_features_from_uniform(
        probX, probE, prob_charges, dense.node_mask, unif)
    X_t = F.one_hot(Xi, num_classes=nm.X_classes)
    E_t = F.one_hot(Ei, num_classes=nm.E_classes)
    charges_t = F.one_hot(Ci, num_classes=nm.charges_classes)

    eps = pos_noise * dense.node_mask.unsqueeze(-1)
    eps = utils.remove_mean_with_mask(eps, dense.node_mask)
    a = nm.get_alpha_bar(t_int=t_int, key="p").unsqueeze(-1)
    s = nm.get_sigma_bar(t_int=t_int, key="p").unsqueeze(-1)
    pos_t = a * dense.pos + s * eps

    return utils.PlaceHolder(X=X_t, charges=charges_t, E=E_t, y=dense.y,
                             pos=pos_t, t_int=t_int, t=t_float,
                             node_mask=dense.node_mask).mask()


def encode_randomness(run_seed, mol_ids, t_int, n_max, device):
    """Per-molecule uniforms and position noise for one (batch, timestep).

    Keyed on (run_seed, mol_id, purpose@t), so molecule i gets the same
    randomness at timestep t in every run and every batching. Reuses
    rng.derive_seed by folding t into the purpose string, so midi/rng.py needs
    no change.
    """
    n_unif = rng.n_unif_per_step(n_max)
    unif = torch.empty(len(mol_ids), n_unif, dtype=torch.float32)
    pos = torch.empty(len(mol_ids), n_max, 3, dtype=torch.float32)
    for b, mid in enumerate(mol_ids):
        g = torch.Generator()
        g.manual_seed(rng.derive_seed(run_seed, mid, f"encode_cat@t{t_int}"))
        unif[b] = torch.rand(n_unif, generator=g)
        g.manual_seed(rng.derive_seed(run_seed, mid, f"encode_pos@t{t_int}"))
        pos[b] = torch.randn(n_max, 3, generator=g)
    return unif.to(device), pos.to(device)


# ---------------------------------------------------------------------------
# Setup shared by all modes
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(path, limit=None):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and f.tell() > limit:
                break
    return h.hexdigest()


def setup(cfg):
    device = torch.device("cuda" if (cfg.general.gpus > 0 and torch.cuda.is_available()) else "cpu")
    datamodule = qm9_dataset.QM9DataModule(cfg)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    model = FullDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only, dataset_infos=dataset_infos, train_smiles=[])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert not model.training, "dropout is 0.1; eval mode is mandatory for reproducibility"
    return device, datamodule, dataset_infos, model


def qm9_root_from_cfg(cfg):
    """The dataset root, resolved the same way QM9DataModule resolves it."""
    import pathlib
    from hydra.utils import get_original_cwd
    base = pathlib.Path(get_original_cwd()).parents[0]
    return str(base / cfg.dataset.datadir)


def timestep_grid(T, every):
    """t_int values to capture: T, T-every, ..., down to >= every."""
    return list(range(T, 0, -every))


# ---------------------------------------------------------------------------
# mode = encode
# ---------------------------------------------------------------------------

def run_encode(cfg, device, datamodule, dataset_infos, model):
    from torch_geometric.data import Batch

    split = cfg.capture.split
    dataset = {"train": datamodule.train_dataloader().dataset,
               "val": datamodule.val_dataloader().dataset,
               "test": datamodule.test_dataloader().dataset}[split]

    store = os.path.abspath(cfg.capture.store)
    os.makedirs(store, exist_ok=True)

    # ---- labels, aligned and verified -------------------------------------
    # Resolve the dataset root exactly as QM9DataModule does, so capture and the
    # datamodule can never disagree about which raw files these labels came from:
    #     base_path = pathlib.Path(get_original_cwd()).parents[0]
    #     root_path = base_path / cfg.dataset.datadir
    datadir_root = cfg.capture.datadir_root or qm9_root_from_cfg(cfg)
    print(f"[labels] building alignment for split={split} from {datadir_root}")
    align = qm9_labels.build_alignment(datadir_root, split)
    print(f"[labels] {align['n_csv']} CSV rows, {align['n_dropped']} uncharacterized dropped, "
          f"{len(align['sdf_index'])} survive")
    qm9_labels.verify_against_dataset(
        align, dataset,
        raw_dir=os.path.join(datadir_root, "raw") if cfg.capture.verify_smiles else None,
        n_check=cfg.capture.smiles_check_n)

    N = int(cfg.capture.n_molecules) if cfg.capture.n_molecules else len(dataset)
    N = min(N, len(dataset))
    mol_ids = list(range(N))        # dataset positions, which are the global ids
    T = model.T
    ts = timestep_grid(T, int(cfg.capture.every))
    L = model.model.n_layers
    bs = int(cfg.capture.batch_size)

    print(f"[grid] N={N}  layers={L}  timesteps={len(ts)} {ts}  features={F_TOTAL}")
    nbytes = L * len(ts) * N * F_TOTAL * 4
    print(f"[grid] store size = {nbytes / 1024**3:.3f} GiB across {L} files")
    print(f"[grid] forward passes = {N * len(ts):,}")

    # ---- allocate ---------------------------------------------------------
    mm = []
    for k in range(L):
        p = os.path.join(store, f"pooled_L{k:02d}.npy")
        mm.append(np.lib.format.open_memmap(
            p, mode="w+", dtype=np.float32, shape=(len(ts), N, F_TOTAL)))

    ctx = CaptureCtx(L)
    handles = [model.model.tf_layers[k].register_forward_hook(make_capture_hook(ctx, k))
               for k in range(L)]

    nm = model.noise_model
    t0 = time.time()
    done = 0
    try:
        with torch.no_grad():
            for lo in range(0, N, bs):
                hi = min(lo + bs, N)
                ids = mol_ids[lo:hi]
                batch = Batch.from_data_list([dataset[j] for j in ids]).to(device)
                dense = utils.to_dense(batch, dataset_infos, device=device)
                dense = pad_dense(dense, N_MAX)

                for ti, t_int in enumerate(ts):
                    unif, pos_noise = encode_randomness(
                        cfg.capture.run_seed, ids, t_int, N_MAX, device)
                    z_t = noise_to_t(nm, dense.copy(), t_int, unif, pos_noise)
                    extra = model.extra_features(z_t)
                    ctx.reset()
                    ctx.enabled = True
                    _ = model.forward(z_t, extra)
                    ctx.enabled = False
                    for k in range(L):
                        f = ctx.buf[k]
                        if f is None:
                            raise RuntimeError(f"layer {k} hook did not fire")
                        arr = f.numpy()
                        if not np.isfinite(arr).all():
                            bad = int((~np.isfinite(arr)).sum())
                            raise RuntimeError(
                                f"{bad} non-finite values at layer {k}, t={t_int}, "
                                f"molecules {lo}:{hi}. Capture aborted rather than "
                                f"storing a poisoned array.")
                        mm[k][ti, lo:hi, :] = arr
                done = hi
                el = time.time() - t0
                rate = done / el * 3600
                print(f"[capture] {done}/{N} molecules  {el/60:.1f} min  "
                      f"{rate:,.0f} mol/GPU-hour  eta {(N-done)/max(rate/3600,1e-9)/60:.1f} min",
                      flush=True)
    finally:
        for h in handles:
            h.remove()
        for a in mm:
            a.flush()

    elapsed = time.time() - t0
    rate = N / elapsed * 3600

    # ---- labels for the captured subset ----------------------------------
    lab = {k: v[:N] for k, v in align["labels"].items()}
    lab["sdf_index"] = align["sdf_index"][:N]
    lab["dataset_pos"] = np.asarray(mol_ids, dtype=np.int64)
    lab["n_atoms"] = np.array([int(dataset[j].x.numel()) for j in mol_ids], dtype=np.int64)
    np.savez(os.path.join(store, "labels.npz"), **lab)

    manifest = {
        "mode": "encode",
        "split": split,
        "n_molecules": N,
        "n_layers": L,
        "timesteps": ts,
        "every": int(cfg.capture.every),
        "T": T,
        "n_max": N_MAX,
        "features": F_TOTAL,
        "feature_slices": FEATURE_SLICES,
        "dead_y_layer": DEAD_Y_LAYER,
        "run_seed": int(cfg.capture.run_seed),
        "batch_size": bs,
        "dtype": "float32",
        "checkpoint": cfg.general.test_only,
        "checkpoint_sha256_first_1mb": _sha256(cfg.general.test_only, limit=1 << 20),
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "elapsed_seconds": elapsed,
        "molecules_per_gpu_hour": rate,
        "store_bytes": nbytes,
        "labels_n_dropped": align["n_dropped"],
        "labels_n_csv": align["n_csv"],
    }
    with open(os.path.join(store, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[done] {N} molecules in {elapsed/60:.1f} min")
    print(f"[done] MEASURED RATE: {rate:,.0f} molecules per GPU-hour")
    print(f"[done] store: {store}  ({nbytes/1024**3:.3f} GiB)")
    print(f"[done] record the rate in the run manifest and in Section 3.11, which "
          f"currently asserts 472 mol/GPU-hour as a projection.")
    return manifest


# ---------------------------------------------------------------------------
# mode = liveness
# ---------------------------------------------------------------------------

def run_liveness(cfg, device, datamodule, dataset_infos, model):
    from torch_geometric.data import Batch
    dataset = datamodule.test_dataloader().dataset
    ids = list(range(int(cfg.capture.liveness_n)))
    batch = Batch.from_data_list([dataset[j] for j in ids]).to(device)
    dense = pad_dense(utils.to_dense(batch, dataset_infos, device=device), N_MAX)
    t_int = int(cfg.capture.liveness_t)
    unif, pos_noise = encode_randomness(cfg.capture.run_seed, ids, t_int, N_MAX, device)
    z_t = noise_to_t(model.noise_model, dense.copy(), t_int, unif, pos_noise)
    extra = model.extra_features(z_t)

    with torch.no_grad():
        base = model.forward(z_t, extra)
        ref = (base.X.clone(), base.E.clone(), base.pos.clone())

    L = model.model.n_layers
    alpha = float(cfg.capture.liveness_alpha)
    rows = []
    print(f"\n[liveness] t={t_int}  alpha={alpha}  {len(ids)} molecules")
    print(f"{'layer':>5} {'chan':>5} {'max|dX|':>12} {'max|dE|':>12} {'max|dpos|':>12}  verdict")
    for k in range(L):
        for ch, dim in (("y", D_Y), ("X", D_X), ("E", D_E)):
            state = {"channel": ch, "alpha": alpha, "v": None}
            g = torch.Generator().manual_seed(rng.derive_seed(0, k, f"liveness_{ch}"))
            v = torch.randn(dim, generator=g)
            state["v"] = (v / v.norm()).to(device)
            h = model.model.tf_layers[k].register_forward_hook(make_inject_hook(state))
            try:
                with torch.no_grad():
                    out = model.forward(z_t, extra)
                d = [float((a - b).abs().max()) for a, b in
                     zip((out.X, out.E, out.pos), ref)]
            finally:
                h.remove()
            dead = max(d) == 0.0
            note = "DEAD" if dead else "live"
            if ch == "y" and k == DEAD_Y_LAYER:
                note += "  (expected dead: mlp_out_y is commented out)"
            rows.append({"layer": k, "channel": ch, "dX": d[0], "dE": d[1],
                         "dpos": d[2], "dead": dead})
            print(f"{k:>5} {ch:>5} {d[0]:>12.3e} {d[1]:>12.3e} {d[2]:>12.3e}  {note}")

    store = os.path.abspath(cfg.capture.store)
    os.makedirs(store, exist_ok=True)
    with open(os.path.join(store, "liveness.json"), "w") as f:
        json.dump(rows, f, indent=2)
    n_dead = sum(r["dead"] for r in rows)
    print(f"\n[liveness] {n_dead}/{len(rows)} sites dead. Written to {store}/liveness.json")
    print("[liveness] a dead site can be probed but never steered: exclude it from "
          "Phase Two injection sites.")
    return rows


# ---------------------------------------------------------------------------
# mode = rotation
# ---------------------------------------------------------------------------

def run_rotation(cfg, device, datamodule, dataset_infos, model):
    """Section 3.6's validation: scalar channels must be rotation-invariant."""
    from torch_geometric.data import Batch
    dataset = datamodule.test_dataloader().dataset
    ids = list(range(int(cfg.capture.liveness_n)))
    batch = Batch.from_data_list([dataset[j] for j in ids]).to(device)
    dense = pad_dense(utils.to_dense(batch, dataset_infos, device=device), N_MAX)
    t_int = int(cfg.capture.liveness_t)
    unif, pos_noise = encode_randomness(cfg.capture.run_seed, ids, t_int, N_MAX, device)

    L = model.model.n_layers
    ctx = CaptureCtx(L)
    handles = [model.model.tf_layers[k].register_forward_hook(make_capture_hook(ctx, k))
               for k in range(L)]

    def capture(rot=None):
        d = dense.copy()
        if rot is not None:
            d.pos = (d.pos @ rot.T) * d.node_mask.unsqueeze(-1)
        pn = pos_noise if rot is None else pos_noise @ rot.T
        z = noise_to_t(model.noise_model, d, t_int, unif, pn)
        ctx.reset(); ctx.enabled = True
        with torch.no_grad():
            model.forward(z, model.extra_features(z))
        ctx.enabled = False
        return [b.clone() for b in ctx.buf]

    try:
        a = capture(None)
        g = torch.Generator().manual_seed(rng.derive_seed(0, 0, "rotation"))
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]                      # proper rotation, det = +1
        b = capture(q.to(device))
    finally:
        for h in handles:
            h.remove()

    ys, xs, es = FEATURE_SLICES["y"], FEATURE_SLICES["X_mean"], FEATURE_SLICES["E_mean"]
    print(f"\n[rotation] scalar-channel invariance, t={t_int}, det(Q)={float(torch.det(q)):+.6f}")
    print(f"{'layer':>5} {'max|dy|':>12} {'max|dX_mean|':>14} {'max|dE_mean|':>14}")
    worst = 0.0
    for k in range(L):
        d = (a[k] - b[k]).abs()
        dy = float(d[:, ys[0]:ys[1]].max())
        dx = float(d[:, xs[0]:xs[1]].max())
        de = float(d[:, es[0]:es[1]].max())
        worst = max(worst, dy, dx, de)
        print(f"{k:>5} {dy:>12.3e} {dx:>14.3e} {de:>14.3e}")
    print(f"\n[rotation] worst deviation {worst:.3e}")
    print("[rotation] float32 forward passes accumulate rounding, so expect ~1e-4 "
          "rather than exact zero. Anything at 1e-2 or above means the capture site "
          "is not invariant and the probe is reading orientation, not structure.")
    return worst


# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: omegaconf.DictConfig):
    assert cfg.dataset.name == "qm9", "capture.py is written for QM9"
    assert cfg.general.test_only, "pass general.test_only=<absolute ckpt path>"
    with omegaconf.open_dict(cfg):
        cfg.capture = omegaconf.OmegaConf.merge(
            omegaconf.OmegaConf.create({
                "mode": "encode",
                "split": "test",
                "store": "activations",
                "datadir_root": None,
                "n_molecules": 2000,
                "every": 50,
                "batch_size": 32,
                "run_seed": 1234,
                "verify_smiles": False,
                "smiles_check_n": None,
                "liveness_n": 8,
                "liveness_t": 400,
                "liveness_alpha": 4.0,
            }),
            cfg.get("capture", {}) or {})

    device, datamodule, dataset_infos, model = setup(cfg)
    print(f"[setup] device={device}  T={model.T}  layers={model.model.n_layers}")
    if device.type == "cuda":
        print(f"[setup] {torch.cuda.get_device_name(0)}")

    mode = cfg.capture.mode
    if mode == "encode":
        run_encode(cfg, device, datamodule, dataset_infos, model)
    elif mode == "liveness":
        run_liveness(cfg, device, datamodule, dataset_infos, model)
    elif mode == "rotation":
        run_rotation(cfg, device, datamodule, dataset_infos, model)
    else:
        raise ValueError(f"unknown capture.mode={mode}")


if __name__ == "__main__":
    main()
