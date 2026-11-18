"""Unpooled (per-node, per-edge) activation capture against the frozen MiDi checkpoint.

This is the twin of midi/capture.py mode=encode. Everything about the model,
the noising tape, the randomness derivation and the label alignment is reused
unchanged by import, so a raw store and a pooled store built with the same
run_seed, split and timestep grid describe the SAME forward passes. That is the
point: the pooled store must be exactly recoverable from the raw one, and
`repool_store` below is the test.

Why a separate mode rather than a flag on encode
------------------------------------------------
Storage. The pooled record is 1408 floats per (molecule, layer, timestep).
The raw record is:

    y        128                      the global graph vector, already unpooled
    X        29 * 256   =   7,424     per-node features at n_max = 29
    E        29 * 29 * 64 = 53,824    per-edge features

so 61,376 floats, which is 43.6x the pooled record, and E alone is 88% of it.
A full-split raw capture with E is about 337 GB. With X and y only it is 46 GB.
Restricting to three layers and two timesteps brings it to 2.3 GB. The config
therefore takes an explicit layer list and defaults keep_E to False.

What the pooling throws away, and why that matters here
-------------------------------------------------------
pool_nodes maps (bs, 29, 256) -> (bs, 1024) as the concatenation of the mean,
min, max and variance over unmasked nodes. The probe reads that summary. The
injection hook in capture.py and steer.py writes a SINGLE vector broadcast to
every unmasked node:

    output.X = output.X + a * v.view(1, 1, -1) * node_mask.unsqueeze(-1)

Adding the same constant c to every node shifts mean, min and max each by
exactly c, and leaves the variance exactly unchanged, because every deviation
from the mean is unchanged. So of the 1024 node-derived features the probe can
read, a uniform broadcast can move 768 of them in a rigidly coupled way and
cannot move the remaining 256 at all. A per-node write, which is what this
store enables, has 29 * 256 degrees of freedom instead of 256 and can move the
variance block independently of the mean block.

Store layout
------------
    <store>/node_L09.npy ... node_LNN.npy    (T_cap, N, 29, 256)  per-node X
    <store>/graph_L09.npy ...                (T_cap, N, 128)      global y
    <store>/edge_L09.npy ...                 (T_cap, N, 29, 29, 64)  if keep_E
    <store>/node_mask.npy                    (N, 29) bool
    <store>/labels.npz                       identical to the pooled store
    <store>/manifest_raw.json

Usage
-----
    cd midi
    python -u capture_raw.py dataset=qm9 dataset.remove_h=False \
        +experiment=qm9_with_h_adaptive general.wandb=disabled \
        general.gpus=1 general.test_only=<abs ckpt path> \
        +capture.split=test +capture.n_molecules=2000 +capture.every=50 \
        +capture.batch_size=32 +capture.run_seed=1234 \
        +capture.layers=[9,10,11] +capture.keep_E=false \
        +capture.store=<abs path>/activations_raw

Then verify against the pooled store built with the same seed and grid:

    python -m midi.capture_raw --repool <raw store> --against <pooled store>
"""

import json
import os
import sys
import time

import numpy as np
import torch

import midi._compat  # noqa: F401  -- torch.load default; must precede any load
import hydra
import omegaconf

from midi import rng, utils
from midi import qm9_labels
from midi.capture import (
    N_MAX, D_Y, D_X, D_E, F_TOTAL, FEATURE_SLICES, DEAD_Y_LAYER,
    CaptureCtx, pool_nodes, pool_edges,
    pad_dense, noise_to_t, encode_randomness, timestep_grid,
    setup, qm9_root_from_cfg, _git_commit, _sha256,
)


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

def make_capture_hook_raw(ctx, slot, keep_E):
    """Store the layer output verbatim. No pooling, no concatenation.

    `slot` is the index into ctx.buf, which indexes the REQUESTED layer list,
    not the model's layer numbering. The manifest records the mapping.

    output is a PlaceHolder with
        output.X  (bs, n_max, 256)
        output.E  (bs, n_max, n_max, 64)
        output.y  (bs, 128)
        output.node_mask (bs, n_max) bool
    """
    def hook(module, inputs, output):
        if not ctx.enabled:
            return None
        rec = {
            "X": output.X.detach().to(torch.float32).cpu(),
            "y": output.y.detach().to(torch.float32).cpu(),
            "node_mask": output.node_mask.detach().cpu(),
        }
        if keep_E:
            rec["E"] = output.E.detach().to(torch.float32).cpu()
        ctx.buf[slot] = rec
        return None
    return hook


# ---------------------------------------------------------------------------
# mode = raw
# ---------------------------------------------------------------------------

def run_raw(cfg, device, datamodule, dataset_infos, model):
    from torch_geometric.data import Batch

    split = cfg.capture.split
    dataset = {"train": datamodule.train_dataloader().dataset,
               "val": datamodule.val_dataloader().dataset,
               "test": datamodule.test_dataloader().dataset}[split]

    store = os.path.abspath(cfg.capture.store)
    os.makedirs(store, exist_ok=True)

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
    mol_ids = list(range(N))
    T = model.T
    ts = timestep_grid(T, int(cfg.capture.every))
    L_model = model.model.n_layers
    bs = int(cfg.capture.batch_size)
    keep_E = bool(cfg.capture.keep_E)

    layers = list(cfg.capture.layers) if cfg.capture.layers else list(range(L_model))
    for k in layers:
        if not (0 <= k < L_model):
            raise ValueError(f"layer {k} outside 0..{L_model - 1}")
    if DEAD_Y_LAYER in layers:
        print(f"[warn] layer {DEAD_Y_LAYER} requested. Its y reaches no output head: "
              f"probe-viable, intervention-dead. Keep it only as a negative control.")

    per_mol = D_Y + N_MAX * D_X + (N_MAX * N_MAX * D_E if keep_E else 0)
    nbytes = len(layers) * len(ts) * N * per_mol * 4
    print(f"[grid] N={N}  layers={layers}  timesteps={len(ts)} {ts}  keep_E={keep_E}")
    print(f"[grid] {per_mol:,} floats per (molecule, layer, timestep), "
          f"{per_mol / F_TOTAL:.1f}x the pooled record")
    print(f"[grid] store size = {nbytes / 1024**3:.2f} GiB")
    print(f"[grid] forward passes = {N * len(ts):,}")
    if nbytes > 60 * 1024**3:
        raise SystemExit(
            f"refusing to write {nbytes / 1024**3:.1f} GiB. Narrow capture.layers, "
            f"capture.every or capture.n_molecules, or set keep_E=false.")

    # ---- allocate ---------------------------------------------------------
    mm_node, mm_graph, mm_edge = [], [], []
    for k in layers:
        mm_node.append(np.lib.format.open_memmap(
            os.path.join(store, f"node_L{k:02d}.npy"), mode="w+",
            dtype=np.float32, shape=(len(ts), N, N_MAX, D_X)))
        mm_graph.append(np.lib.format.open_memmap(
            os.path.join(store, f"graph_L{k:02d}.npy"), mode="w+",
            dtype=np.float32, shape=(len(ts), N, D_Y)))
        if keep_E:
            mm_edge.append(np.lib.format.open_memmap(
                os.path.join(store, f"edge_L{k:02d}.npy"), mode="w+",
                dtype=np.float32, shape=(len(ts), N, N_MAX, N_MAX, D_E)))
    mask_out = np.lib.format.open_memmap(
        os.path.join(store, "node_mask.npy"), mode="w+", dtype=bool, shape=(N, N_MAX))

    ctx = CaptureCtx(len(layers))
    handles = [model.model.tf_layers[k].register_forward_hook(
        make_capture_hook_raw(ctx, slot, keep_E)) for slot, k in enumerate(layers)]

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

                    for slot, k in enumerate(layers):
                        rec = ctx.buf[slot]
                        if rec is None:
                            raise RuntimeError(f"layer {k} hook did not fire")
                        Xa = rec["X"].numpy()
                        ya = rec["y"].numpy()
                        if not (np.isfinite(Xa).all() and np.isfinite(ya).all()):
                            raise RuntimeError(
                                f"non-finite values at layer {k}, t={t_int}, "
                                f"molecules {lo}:{hi}. Capture aborted rather than "
                                f"storing a poisoned array.")
                        mm_node[slot][ti, lo:hi] = Xa
                        mm_graph[slot][ti, lo:hi] = ya
                        if keep_E:
                            Ea = rec["E"].numpy()
                            if not np.isfinite(Ea).all():
                                raise RuntimeError(f"non-finite E at layer {k}, t={t_int}")
                            mm_edge[slot][ti, lo:hi] = Ea
                    if ti == 0:
                        mask_out[lo:hi] = ctx.buf[0]["node_mask"].numpy()

                done = hi
                el = time.time() - t0
                rate = done / el * 3600
                print(f"[raw] {done}/{N} molecules  {el/60:.1f} min  "
                      f"{rate:,.0f} mol/GPU-hour  eta {(N-done)/max(rate/3600,1e-9)/60:.1f} min",
                      flush=True)
    finally:
        for h in handles:
            h.remove()
        for a in mm_node + mm_graph + mm_edge + [mask_out]:
            a.flush()

    elapsed = time.time() - t0

    lab = {k: v[:N] for k, v in align["labels"].items()}
    lab["sdf_index"] = align["sdf_index"][:N]
    lab["dataset_pos"] = np.asarray(mol_ids, dtype=np.int64)
    lab["n_atoms"] = np.array([int(dataset[j].x.numel()) for j in mol_ids], dtype=np.int64)
    comp = qm9_labels.composition_counts(dataset, mol_ids, dataset_infos.atom_decoder)
    print("[labels] composition:", qm9_labels.verify_composition(comp, lab["n_atoms"]))
    lab.update(comp)
    np.savez(os.path.join(store, "labels.npz"), **lab)

    manifest = {
        "mode": "raw",
        "split": split,
        "n_molecules": N,
        "layers": layers,
        "slot_to_layer": {str(s): k for s, k in enumerate(layers)},
        "timesteps": ts,
        "every": int(cfg.capture.every),
        "T": T,
        "n_max": N_MAX,
        "d_y": D_Y, "d_x": D_X, "d_e": D_E,
        "keep_E": keep_E,
        "pooled_feature_slices": FEATURE_SLICES,
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
        "store_bytes": nbytes,
        "labels_n_dropped": align["n_dropped"],
        "labels_n_csv": align["n_csv"],
    }
    with open(os.path.join(store, "manifest_raw.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] {elapsed/60:.1f} min, {nbytes / 1024**3:.2f} GiB -> {store}")


# ---------------------------------------------------------------------------
# Re-pooling: the raw store must reproduce the pooled store exactly
# ---------------------------------------------------------------------------

def repool_layer(raw_store, layer, keep_E=None):
    """Rebuild the 1408-wide pooled record for one layer from the raw store.

    Returns (T_cap, N, 1408) float32, laid out exactly as FEATURE_SLICES says.
    The E blocks are zero when the raw store was written with keep_E=false;
    the node and y blocks are always exact.
    """
    man = json.load(open(os.path.join(raw_store, "manifest_raw.json")))
    if keep_E is None:
        keep_E = man["keep_E"]
    Xn = np.load(os.path.join(raw_store, f"node_L{layer:02d}.npy"), mmap_mode="r")
    yn = np.load(os.path.join(raw_store, f"graph_L{layer:02d}.npy"), mmap_mode="r")
    mask = np.load(os.path.join(raw_store, "node_mask.npy"))
    En = (np.load(os.path.join(raw_store, f"edge_L{layer:02d}.npy"), mmap_mode="r")
          if keep_E else None)

    Tc, N = Xn.shape[0], Xn.shape[1]
    out = np.zeros((Tc, N, F_TOTAL), dtype=np.float32)
    nm = torch.from_numpy(mask)
    for ti in range(Tc):
        X = torch.from_numpy(np.asarray(Xn[ti]))
        blocks = [torch.from_numpy(np.asarray(yn[ti])), pool_nodes(X, nm)]
        if keep_E:
            blocks.append(pool_edges(torch.from_numpy(np.asarray(En[ti])), nm))
        else:
            blocks.append(torch.zeros(N, 4 * D_E))
        out[ti] = torch.cat(blocks, dim=-1).numpy()
    return out


def repool_store(raw_store, pooled_store=None, atol=1e-4, rtol=1e-4):
    """Round-trip gate. Rebuild each captured layer and, if a pooled store built
    with the same seed and grid is given, require agreement on y and the X blocks.

    Agreement is the proof that the raw store is the same forward pass, not a
    different one. A mismatch means the seed, split, timestep grid or checkpoint
    differs, and nothing downstream of the raw store can be trusted until it is
    resolved.
    """
    man = json.load(open(os.path.join(raw_store, "manifest_raw.json")))
    ok = True
    for k in man["layers"]:
        got = repool_layer(raw_store, k)
        print(f"[repool] layer {k}: {got.shape} rebuilt")
        if pooled_store is None:
            continue
        ref = np.load(os.path.join(pooled_store, f"pooled_L{k:02d}.npy"), mmap_mode="r")
        if ref.shape[0] != got.shape[0] or ref.shape[1] < got.shape[1]:
            print(f"  SKIP: pooled store is {ref.shape}, raw rebuild is {got.shape}")
            ok = False
            continue
        ref = np.asarray(ref[:, :got.shape[1], :])
        lo_y, hi_y = FEATURE_SLICES["y"]
        lo_x = FEATURE_SLICES["X_mean"][0]
        hi_x = FEATURE_SLICES["X_var"][1]
        for name, lo, hi in (("y", lo_y, hi_y), ("X blocks", lo_x, hi_x)):
            d = np.abs(got[..., lo:hi] - ref[..., lo:hi])
            close = np.allclose(got[..., lo:hi], ref[..., lo:hi], atol=atol, rtol=rtol)
            print(f"  {name:9s} max abs diff {d.max():.3e}  "
                  f"{'MATCH' if close else 'MISMATCH'}")
            ok = ok and close
        if man["keep_E"]:
            lo_e = FEATURE_SLICES["E_mean"][0]
            d = np.abs(got[..., lo_e:] - ref[..., lo_e:])
            close = np.allclose(got[..., lo_e:], ref[..., lo_e:], atol=atol, rtol=rtol)
            print(f"  E blocks  max abs diff {d.max():.3e}  "
                  f"{'MATCH' if close else 'MISMATCH'}")
            ok = ok and close
    return ok


# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: omegaconf.DictConfig):
    assert cfg.dataset.name == "qm9", "capture_raw.py is written for QM9"
    assert cfg.general.test_only, "pass general.test_only=<absolute ckpt path>"
    with omegaconf.open_dict(cfg):
        cfg.capture = omegaconf.OmegaConf.merge(
            omegaconf.OmegaConf.create({
                "mode": "raw",
                "split": "test",
                "store": "activations_raw",
                "datadir_root": None,
                "n_molecules": 2000,
                "every": 50,
                "batch_size": 32,
                "run_seed": 1234,
                "verify_smiles": False,
                "smiles_check_n": None,
                "layers": [9, 10, 11],
                "keep_E": False,
            }),
            cfg.get("capture", {}) or {})

    device, datamodule, dataset_infos, model = setup(cfg)
    print(f"[setup] device={device}  T={model.T}  layers={model.model.n_layers}")
    if device.type == "cuda":
        print(f"[setup] {torch.cuda.get_device_name(0)}")
    run_raw(cfg, device, datamodule, dataset_infos, model)


if __name__ == "__main__":
    if "--repool" in sys.argv:
        i = sys.argv.index("--repool")
        raw = sys.argv[i + 1]
        ref = None
        if "--against" in sys.argv:
            ref = sys.argv[sys.argv.index("--against") + 1]
        sys.exit(0 if repool_store(raw, ref) else 1)
    main()
