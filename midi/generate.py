"""Chunked, resumable baseline generation with CSV progress logging.

Why this exists
---------------
`main.py` generates the whole target in one `trainer.test` call: every molecule
is held in memory and nothing is written until the very end. On 2026-11-09 a
run of 10,000 molecules was killed by a system shutdown after 31 h 23 m
(Kernel-Power event 109) and produced **nothing** -- no molecules, no metrics,
and not even a flushed log, because Python's stdout buffer died with the
process.

This script removes that failure mode:

  * generation proceeds in fixed-size CHUNKS; each chunk is written to disk
    atomically before the next begins, so a kill costs at most one chunk;
  * a CSV row is appended per chunk, so progress, throughput and ETA are
    readable while the job runs;
  * a re-run RESUMES from the chunks already on disk;
  * GPU memory is released between chunks, which also lowers the OOM risk on a
    6 GB card.

Resume is EXACT, not approximate. Every molecule's randomness is keyed on
(run_seed, molecule index) by midi/rng.py, so molecule i is identical whether
it was produced in the first pass or after three restarts. That only holds if
the chunk size is unchanged, because chunk size fixes batch composition and
positions drift ~2e-5 Angstrom when composition differs (see test_pairing.py).
The script therefore refuses to resume with a different chunk size or run seed.

Usage
-----
    # generate (default); safe to re-run after any interruption
    python -u generate.py dataset=qm9 dataset.remove_h=False \
      +experiment=qm9_with_h_adaptive general.wandb=disabled general.gpus=1 \
      "general.test_only=<abs path to ckpt>" \
      +gen.target=10000 +gen.chunk=250 +gen.out=<abs path to output dir>

    # metrics only, over whatever chunks are on disk
    ... +gen.mode=metrics

Options under `+gen.`:
    mode            generate | metrics          (default generate)
    target          molecules to produce        (default 10000)
    chunk           molecules per chunk         (default 250)
    out             output directory            (default <repo>/baseline_run)
    run_seed        RNG key                     (default 1234)
    metrics_at_end  run metrics once complete   (default true)

Outputs in `out/`:
    chunk_00000.pt      per-chunk tensors (atom types, charges, bonds, positions)
    progress.csv        one row per completed chunk
    state.json          run identity and progress
"""

import csv
import datetime
import json
import os
import time

import torch

import midi._compat  # noqa: F401  -- torch.load default, must precede any load
import hydra
import omegaconf

from midi import rng
from midi.analysis.rdkit_functions import Molecule
from midi.datasets import qm9_dataset
from midi.diffusion_model import FullDenoisingDiffusion

N_MAX = 29          # QM9 with explicit hydrogens; global, never the batch max

CSV_FIELDS = [
    "timestamp_utc", "chunk_index", "mol_from", "mol_to", "n_molecules",
    "chunk_seconds", "cumulative_seconds",
    "chunk_mol_per_hour", "cumulative_mol_per_hour",
    "molecules_done", "target", "pct_complete", "eta_hours",
    "rdkit_valid", "rdkit_valid_pct", "mean_atoms",
    "gpu_alloc_mb", "gpu_reserved_mb",
]


# ---------------------------------------------------------------- helpers ---

def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _atomic_save(obj, path, torch_save=False):
    """Write via a temp file then rename, so a kill never leaves a torn file.

    This is what makes "at most one chunk is lost" true rather than hopeful: any
    chunk_*.pt present on disk is complete by construction.
    """
    tmp = path + ".tmp"
    if torch_save:
        torch.save(obj, tmp)
    else:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _chunk_path(out, i):
    return os.path.join(out, f"chunk_{i:05d}.pt")


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())      # survive a hard power loss, not just a kill


def _save_chunk(path, mols, mol_ids):
    _atomic_save({
        "mol_ids": list(mol_ids),
        "atom_types": [m.atom_types.cpu() for m in mols],
        "charges": [m.charges.cpu() for m in mols],
        "bond_types": [m.bond_types.cpu() for m in mols],
        "positions": [m.positions.cpu() for m in mols],
    }, path, torch_save=True)


def _load_chunk(path, atom_decoder, map_location="cpu"):
    d = torch.load(path, map_location=map_location)
    return [Molecule(atom_types=d["atom_types"][i], charges=d["charges"][i],
                     bond_types=d["bond_types"][i], positions=d["positions"][i],
                     atom_decoder=atom_decoder)
            for i in range(len(d["atom_types"]))]


def _count_valid(mols):
    """Cheap RDKit validity, for early warning only.

    NOT the reported metric. The figure that goes beside MiDi's Table 2 comes
    from SamplingMetrics in `mode=metrics`, which applies the MOSES conventions.
    This is here so a collapse in validity is visible in the CSV within one
    chunk instead of at the end of the run.
    """
    from rdkit import Chem
    ok = 0
    for m in mols:
        try:
            raw = getattr(m, "rdkit_mol", None)
            if raw is None:
                continue
            probe = Chem.Mol(raw)          # copy: SanitizeMol mutates
            Chem.SanitizeMol(probe)
            if Chem.MolToSmiles(probe):
                ok += 1
        except Exception:
            pass
    return ok


def _gpu_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.memory_reserved() / 1024 ** 2)


def _setup(cfg):
    device = torch.device("cuda" if (cfg.general.gpus > 0 and torch.cuda.is_available()) else "cpu")
    datamodule = qm9_dataset.QM9DataModule(cfg)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    # Real training SMILES: novelty is "absent from the training set", so an
    # empty list would report 100% novelty for everything.
    train_smiles = list(datamodule.train_dataloader().dataset.smiles)
    model = FullDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only, dataset_infos=dataset_infos, train_smiles=train_smiles)
    model = model.to(device).eval()
    assert not model.training, "eval mode is mandatory: dropout is 0.1"
    return device, dataset_infos, model, len(train_smiles)


# --------------------------------------------------------------- generate ---

def run_generate(cfg, device, dataset_infos, model, n_train_smiles):
    out = os.path.abspath(cfg.gen.out)
    os.makedirs(out, exist_ok=True)
    target = int(cfg.gen.target)
    chunk = int(cfg.gen.chunk)
    seed = int(cfg.gen.run_seed)
    n_chunks = (target + chunk - 1) // chunk
    state_path = os.path.join(out, "state.json")
    csv_path = os.path.join(out, "progress.csv")

    # ---- resume guard --------------------------------------------------
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                prev = json.load(f)
        except Exception:
            prev = {}
        for key, now in (("chunk", chunk), ("run_seed", seed)):
            was = prev.get(key)
            if was is not None and was != now:
                raise SystemExit(
                    f"refusing to resume: {key} was {was}, now {now}.\n"
                    f"Chunk size and run seed fix which molecules land in which\n"
                    f"batch, and batch composition changes positions by ~2e-5\n"
                    f"Angstrom. Either restore {key}={was} or generate into a\n"
                    f"fresh +gen.out directory.")

    # Trust the files, not the state file: an atomically-written chunk is
    # complete, so the directory is the authoritative record of progress.
    done = {i for i in range(n_chunks) if os.path.exists(_chunk_path(out, i))}
    todo = [i for i in range(n_chunks) if i not in done]

    alloc, reserved = _gpu_mb()
    print(f"[gen] out        {out}")
    print(f"[gen] target     {target} molecules in {n_chunks} chunks of {chunk}")
    print(f"[gen] run_seed   {seed}   n_max {N_MAX}   T {model.T}")
    print(f"[gen] device     {device}"
          + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    print(f"[gen] train smiles for novelty: {n_train_smiles:,}")
    if done:
        print(f"[gen] RESUMING: {len(done)}/{n_chunks} chunks already on disk "
              f"({len(done) * chunk:,} molecules); {len(todo)} to go")
    else:
        print(f"[gen] fresh run")
    if not todo:
        print("[gen] nothing to generate; all chunks present")
        return out, n_chunks

    cum_prev = len(done) * chunk
    t_run = time.time()
    produced = 0
    try:
        for ci in todo:
            lo = ci * chunk
            hi = min(lo + chunk, target)
            ids = list(range(lo, hi))
            t0 = time.time()

            n_nodes = rng.sample_n_nodes(seed, ids, model.node_dist.prob)
            with torch.no_grad():
                mols = model.sample_batch(
                    n_nodes=n_nodes.tolist(), mol_ids=ids, run_seed=seed,
                    n_max_fixed=N_MAX, test=True, save_final=0, keep_chain=0,
                    visualize=False, batch_id=ci)

            _save_chunk(_chunk_path(out, ci), mols, ids)

            dt = time.time() - t0
            produced += len(mols)
            cum_sec = time.time() - t_run
            molecules_done = cum_prev + produced
            cum_rate = produced / cum_sec * 3600 if cum_sec > 0 else 0.0
            remaining = target - molecules_done
            n_valid = _count_valid(mols)
            mean_atoms = sum(int(m.num_nodes) for m in mols) / max(len(mols), 1)
            alloc, reserved = _gpu_mb()

            _append_csv(csv_path, {
                "timestamp_utc": _now(),
                "chunk_index": ci,
                "mol_from": lo,
                "mol_to": hi - 1,
                "n_molecules": len(mols),
                "chunk_seconds": round(dt, 2),
                "cumulative_seconds": round(cum_sec, 2),
                "chunk_mol_per_hour": round(len(mols) / dt * 3600, 1) if dt > 0 else "",
                "cumulative_mol_per_hour": round(cum_rate, 1),
                "molecules_done": molecules_done,
                "target": target,
                "pct_complete": round(100.0 * molecules_done / target, 2),
                "eta_hours": round(remaining / cum_rate, 2) if cum_rate > 0 else "",
                "rdkit_valid": n_valid,
                "rdkit_valid_pct": round(100.0 * n_valid / max(len(mols), 1), 2),
                "mean_atoms": round(mean_atoms, 2),
                "gpu_alloc_mb": round(alloc, 1),
                "gpu_reserved_mb": round(reserved, 1),
            })

            _atomic_save({
                "run_seed": seed, "target": target, "chunk": chunk,
                "n_max": N_MAX, "T": model.T, "n_chunks": n_chunks,
                "chunks_done": len(done) + produced // chunk,
                "molecules_done": molecules_done,
                "updated_utc": _now(),
                "checkpoint": str(cfg.general.test_only),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            }, state_path)

            eta = f"{remaining / cum_rate:.2f} h" if cum_rate > 0 else "?"
            print(f"[gen] chunk {ci + 1}/{n_chunks}  mols {lo}-{hi - 1}  "
                  f"{dt:.1f}s  {len(mols) / dt * 3600:,.0f} mol/h  "
                  f"total {molecules_done:,}/{target:,} "
                  f"({100.0 * molecules_done / target:.1f}%)  "
                  f"valid~{100.0 * n_valid / max(len(mols), 1):.1f}%  eta {eta}",
                  flush=True)

            del mols
            if device.type == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print(f"\n[gen] interrupted. {len(done) + produced // chunk} chunks are on "
              f"disk and complete; re-run the same command to resume.", flush=True)
        raise SystemExit(130)

    print(f"\n[gen] generation complete: {cum_prev + produced:,} molecules in "
          f"{(time.time() - t_run) / 3600:.2f} h this pass")
    return out, n_chunks


# ---------------------------------------------------------------- metrics ---

def run_metrics(cfg, device, dataset_infos, model, out=None, n_chunks=None):
    out = os.path.abspath(out or cfg.gen.out)
    files = sorted(f for f in os.listdir(out)
                   if f.startswith("chunk_") and f.endswith(".pt"))
    if not files:
        raise SystemExit(f"no chunk_*.pt in {out}")
    # ---- device: reproduce upstream's configuration exactly ------------
    # SamplingMetrics is device-sensitive in two places that must agree:
    #
    #   * check_stability (analysis/rdkit_functions.py:91) reads
    #     `molecule.atom_types.device` and returns its mol_stable /
    #     n_stable_bonds tensors on THAT device.
    #   * the MeanMetric accumulators inside test_sampling_metrics follow
    #     the LightningModule, so _setup's `model.to(device)` put them on
    #     cuda:0.
    #
    # _save_chunk writes every tensor with .cpu() (it has to: the molecules
    # come off the GPU), so loading them back with map_location="cpu" fed
    # CPU values into CUDA accumulators and torchmetrics raised
    # "Encountered different devices in metric calculation".
    #
    # Upstream hands these metrics GPU molecules (diffusion_model.py:446
    # slices atom_types/E/pos straight out of the sampled batch), and that
    # configuration does run -- zeros_like(stat.atom_types) indexed by a
    # 0-dim CUDA index is a case PyTorch handles, verified directly.
    #
    # But it is unusably slow for a whole store. build_molecule and
    # check_stability call .item() once per atom and once per bond, so each
    # molecule costs ~40 GPU->CPU synchronisations: measured at one pegged
    # core with the GPU at 6% and 10,000 molecules still not loaded after
    # 25 minutes. Upstream never notices because it builds one 250-molecule
    # batch at a time, where sampling dominates.
    #
    # So the default is CPU, and nothing is given up by it: the outputs are
    # integer stability counts, RDKit SMILES (built from .item() values on
    # the CPU either way), integer histograms, and 1-D Wasserstein sums
    # over a few hundred bins. Only float32 rounding in the last bits could
    # differ, and these are reported to three decimals. The statistics
    # tensors are CPU-resident anyway, so CPU is the configuration in which
    # every function here agrees.
    #
    # +gen.metrics_device=cuda restores upstream's placement if a
    # like-for-like device comparison is ever wanted.
    mdev = str(cfg.gen.get("metrics_device", "") or device)
    print(f"\n[metrics] loading {len(files)} chunks from {out}")
    print(f"[metrics] device {mdev} (molecules and metric states together)")
    samples = []
    for f in files:
        samples.extend(_load_chunk(os.path.join(out, f),
                                   dataset_infos.atom_decoder,
                                   map_location=mdev))
    model.test_sampling_metrics.to(mdev)
    print(f"[metrics] {len(samples):,} molecules reconstructed")
    if n_chunks is not None and len(files) < n_chunks:
        print(f"[metrics] WARNING: {len(files)}/{n_chunks} chunks present. These "
              f"metrics describe a partial run; say so wherever they are reported.")
    print(f"[metrics] running SamplingMetrics (this is the figure for Table 2)\n")
    model.test_sampling_metrics(samples, model.name, model.current_epoch, model.local_rank)
    print(f"\n[metrics] done. Validity, uniqueness, novelty and the W1 distances "
          f"are printed above.")


# ------------------------------------------------------------------- main ---

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: omegaconf.DictConfig):
    assert cfg.dataset.name == "qm9", "generate.py is written for QM9"
    assert cfg.general.test_only, "pass general.test_only=<absolute ckpt path>"

    import pathlib
    from hydra.utils import get_original_cwd
    default_out = str(pathlib.Path(get_original_cwd()).parents[0] / "baseline_run")

    with omegaconf.open_dict(cfg):
        cfg.gen = omegaconf.OmegaConf.merge(
            omegaconf.OmegaConf.create({
                "mode": "generate",
                "target": 10000,
                "chunk": 250,
                "out": default_out,
                "run_seed": 1234,
                "metrics_at_end": True,
                # Where SamplingMetrics runs. "cpu" is the default and is
                # measured to be far faster here; "cuda" reproduces
                # upstream's device placement. See run_metrics().
                "metrics_device": "cpu",
            }),
            cfg.get("gen", {}) or {})

    device, dataset_infos, model, n_train = _setup(cfg)

    if cfg.gen.mode == "metrics":
        run_metrics(cfg, device, dataset_infos, model)
        return

    out, n_chunks = run_generate(cfg, device, dataset_infos, model, n_train)

    present = len([f for f in os.listdir(out)
                   if f.startswith("chunk_") and f.endswith(".pt")])
    if cfg.gen.metrics_at_end and present >= n_chunks:
        run_metrics(cfg, device, dataset_infos, model, out=out, n_chunks=n_chunks)
    elif cfg.gen.metrics_at_end:
        print(f"[gen] {present}/{n_chunks} chunks present; skipping metrics. "
              f"Re-run to finish, or force with +gen.mode=metrics.")


if __name__ == "__main__":
    main()
