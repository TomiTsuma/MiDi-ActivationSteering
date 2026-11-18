"""Test T3: end-to-end pairing. The gate for Block 1.

What this test must distinguish
-------------------------------
T1 and T2 (tests/test_determinism.py) prove the tape itself is batch
independent, bit for bit. T3 asks the question that actually matters: does
molecule i come out the same when it was batched differently?

The honest answer has two parts, and an earlier version of this test conflated
them:

  * Everything DISCRETE must be bit-identical: the atom count, atom types,
    formal charges and bond types. These come straight from the tape through an
    inverse-CDF lookup, so any difference at all means the randomness is still
    batch dependent. This is the real gate.

  * POSITIONS cannot be bit-identical across different batch shapes, and that
    is not a bug. They are float32 and the sampler applies 500 sequential
    updates through 12 layers. BLAS picks different blocking and threading for
    a batch of 8 than for a batch of 2, so reductions happen in a different
    order; each op differs at ~1e-7 and the difference compounds. Observed
    on CPU: ~2e-5 Angstrom. For scale, MiDi's own Table 2 reports bond lengths
    to 1e-2 Angstrom, so 2e-5 is four orders of magnitude below anything the
    chemistry can see.

Three runs, so the cause is never ambiguous
-------------------------------------------
    A   ids 0..7 in one batch of 8
    A2  ids 0..7 in one batch of 8, again, same order
    B   ids 4,5  in one batch of 2

    C1  A vs A2   same composition. Discrete must be bit-exact; positions must
                  be bit-exact on CPU, within tolerance on CUDA. This isolates
                  RNG determinism from batch-shape effects.
    C2  A vs B    discrete must be bit-exact. THIS IS THE GATE.
    C3  A vs B    positions within --pos-tol. A breach is reported as a
                  constraint on the experiment, not as a pairing bug.

If C1 passes and C2 passes, the pairing is sound: generate the steered and
unsteered arms with identical batch composition and they pair exactly on
every discrete channel, with positions agreeing to float precision.

Run it like main.py, from inside midi/ after `pip install -e .`:

    python test_pairing.py dataset=qm9 dataset.remove_h=False \
      +experiment=qm9_with_h_adaptive general.wandb=disabled general.gpus=1 \
      "general.test_only=C:/.../checkpoint_qm9_h_adaptive.ckpt"

Optional: +pairing.pos_tol=1e-3
"""

import torch
import midi._compat  # noqa: F401  -- torch.load default, must precede any load
import hydra
import omegaconf

from midi import rng
from midi.datasets import qm9_dataset
from midi.diffusion_model import FullDenoisingDiffusion

N_MAX = 29          # QM9 with explicit hydrogens. Global, never the batch max.
RUN_SEED = 1234
IDS_A = [0, 1, 2, 3, 4, 5, 6, 7]
IDS_B = [4, 5]      # same molecules, different batch size and position

DISCRETE = ("atom_types", "charges", "bond_types")


def _fields(mol):
    return {"atom_types": mol.atom_types, "charges": mol.charges,
            "bond_types": mol.bond_types, "positions": mol.positions}


def _dev(a, b):
    """Max absolute deviation between two tensors, or None on a shape mismatch."""
    if a.shape != b.shape:
        return None
    if a.is_floating_point():
        return float((a.double() - b.double()).abs().max())
    return float((a != b).sum())


def compare(a, b, label, pos_tol, pos_must_be_exact):
    """Returns (discrete_ok, pos_ok, pos_dev, lines)."""
    fa, fb = _fields(a), _fields(b)
    lines, discrete_ok, pos_ok, pos_dev = [], True, True, None

    for name in DISCRETE:
        d = _dev(fa[name], fb[name])
        if d is None:
            discrete_ok = False
            lines.append(f"    {name:11s} SHAPE MISMATCH "
                         f"{tuple(fa[name].shape)} vs {tuple(fb[name].shape)}")
        elif d == 0:
            lines.append(f"    {name:11s} identical")
        else:
            discrete_ok = False
            lines.append(f"    {name:11s} DIFFERS in {int(d)} entries")

    pos_dev = _dev(fa["positions"], fb["positions"])
    if pos_dev is None:
        pos_ok = False
        lines.append(f"    {'positions':11s} SHAPE MISMATCH")
    elif pos_must_be_exact:
        pos_ok = (pos_dev == 0.0)
        lines.append(f"    {'positions':11s} max abs {pos_dev:.3e}"
                     f"  (must be exactly 0 here)" if not pos_ok
                     else f"    {'positions':11s} identical")
    else:
        pos_ok = (pos_dev <= pos_tol)
        lines.append(f"    {'positions':11s} max abs {pos_dev:.3e}"
                     f"  ({'within' if pos_ok else 'EXCEEDS'} tol {pos_tol:.1e})")
    return discrete_ok, pos_ok, pos_dev, lines


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: omegaconf.DictConfig):
    assert cfg.dataset.name == "qm9", "T3 is written for QM9"
    assert cfg.general.test_only, "pass general.test_only=<absolute ckpt path>"
    pos_tol = float(cfg.get("pairing", {}).get("pos_tol", 1e-3)) \
        if cfg.get("pairing", None) else 1e-3

    use_cuda = bool(cfg.general.gpus > 0 and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    datamodule = qm9_dataset.QM9DataModule(cfg)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

    model = FullDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only, dataset_infos=dataset_infos, train_smiles=[])
    model = model.to(device).eval()
    assert not model.training, \
        "model must be in eval mode: dropout is 0.1 and would break reproducibility"

    T = model.T
    print(f"device={device}  T={T}  n_max={N_MAX}  run_seed={RUN_SEED}  "
          f"pos_tol={pos_tol:.1e}")
    if use_cuda:
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"tape for a batch of {len(IDS_A)}: "
          f"{rng.tape_bytes(len(IDS_A), T, N_MAX) / 1024**2:.1f} MiB\n")

    nA = rng.sample_n_nodes(RUN_SEED, IDS_A, model.node_dist.prob)
    nB = rng.sample_n_nodes(RUN_SEED, IDS_B, model.node_dist.prob)
    print("atom counts, run A:", nA.tolist())
    print("atom counts, run B:", nB.tolist())
    if not (nA[4].item() == nB[0].item() and nA[5].item() == nB[1].item()):
        print("\nT3 FAIL: atom count is not batch-independent. sample_n_nodes was "
              "bypassed somewhere, or node_dist was consumed from the global stream.")
        return 1
    assert int(nA.max()) <= N_MAX, f"an atom count exceeds n_max={N_MAX}"
    print("  atom counts agree\n")

    common = dict(test=True, save_final=0, keep_chain=0, visualize=False,
                  run_seed=RUN_SEED, n_max_fixed=N_MAX)

    with torch.no_grad():
        print(f"run A : ids {IDS_A} in one batch of {len(IDS_A)}")
        molsA = model.sample_batch(n_nodes=nA.tolist(), mol_ids=IDS_A, **common)
        print(f"run A2: ids {IDS_A} again, same composition (RNG determinism control)")
        molsA2 = model.sample_batch(n_nodes=nA.tolist(), mol_ids=IDS_A, **common)
        print(f"run B : ids {IDS_B} in one batch of {len(IDS_B)}")
        molsB = model.sample_batch(n_nodes=nB.tolist(), mol_ids=IDS_B, **common)

    byA = dict(zip(IDS_A, molsA))
    byA2 = dict(zip(IDS_A, molsA2))
    byB = dict(zip(IDS_B, molsB))

    # ---- C1: same composition twice ------------------------------------
    print("\n" + "=" * 66)
    print("C1  same batch composition twice (RNG determinism)")
    c1_discrete, c1_pos, c1_worst = True, True, 0.0
    for mid in IDS_A:
        d_ok, p_ok, dev, lines = compare(
            byA[mid], byA2[mid], mid, pos_tol, pos_must_be_exact=not use_cuda)
        c1_discrete &= d_ok
        c1_pos &= p_ok
        if dev is not None:
            c1_worst = max(c1_worst, dev)
        if not (d_ok and p_ok):
            print(f"  molecule {mid}: DIFFERS")
            for ln in lines:
                print(ln)
    print(f"  discrete bit-exact across all {len(IDS_A)} molecules: {c1_discrete}")
    print(f"  worst position deviation: {c1_worst:.3e}"
          f"  ({'must be 0 on CPU' if not use_cuda else f'tol {pos_tol:.1e} on CUDA'})")

    # ---- C2 and C3: different composition -------------------------------
    print("\n" + "=" * 66)
    print("C2/C3  different batch composition (the pairing gate)")
    c2_ok, c3_ok, c3_worst = True, True, 0.0
    for mid in IDS_B:
        d_ok, p_ok, dev, lines = compare(
            byA[mid], byB[mid], mid, pos_tol, pos_must_be_exact=False)
        c2_ok &= d_ok
        c3_ok &= p_ok
        if dev is not None:
            c3_worst = max(c3_worst, dev)
        print(f"  molecule {mid}  ({byA[mid].num_nodes} atoms)")
        for ln in lines:
            print(ln)
    print("=" * 66)

    # ---- verdict ---------------------------------------------------------
    print()
    if not c1_discrete:
        print("T3 FAIL: the same batch run twice gave different discrete output.")
        print("The randomness is not determined by (run_seed, mol_id) at all.")
        print("Look for a torch.randn, .multinomial or .sample() still reading")
        print("the global stream inside sample_limit_dist or")
        print("sample_zs_from_zt_and_pred.")
        return 1
    if not c1_pos:
        print(f"T3 FAIL: identical batches gave positions differing by {c1_worst:.3e}.")
        print("On CPU this must be exactly zero. Non-determinism inside the")
        print("forward pass itself, not the tape.")
        return 1
    if not c2_ok:
        print("T3 FAIL: a discrete channel changed with batch composition.")
        print("This is the real pairing bug. Check that n_max_fixed=29 is being")
        print("honoured (a batch-dependent n_max changes the uniform budget and")
        print("so the draw sequence), and that nothing re-seeds per batch.")
        return 1

    print("T3 PASS.")
    print(f"  Discrete channels (atom count, atom types, charges, bond types) are")
    print(f"  bit-identical regardless of batch composition. Positions agree to")
    print(f"  {c3_worst:.3e} Angstrom across different batch shapes, which is")
    print(f"  float32 accumulation over {T} sequential steps, not a pairing fault:")
    print(f"  MiDi's own Table 2 reports bond lengths to 1e-2 Angstrom.")
    if not c3_ok:
        print()
        print(f"  WARNING: that deviation exceeds the {pos_tol:.1e} tolerance.")
        print(f"  The pairing is still sound, but do not compare arms generated")
        print(f"  at different batch sizes. Record this in Section 3.12 as a")
        print(f"  constraint: every arm of a matched pair must be generated with")
        print(f"  identical batch composition and order.")
    else:
        print()
        print(f"  Constraint for Section 3.12: generate both arms of a matched")
        print(f"  pair with identical batch composition and order. Positions drift")
        print(f"  at ~{c3_worst:.0e} if batch shape differs, which is harmless for")
        print(f"  chemistry but makes bit-level comparison meaningless.")
    return 0


if __name__ == "__main__":
    main()
