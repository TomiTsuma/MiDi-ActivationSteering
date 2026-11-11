"""Backfill element counts into an existing activation store's labels.npz.

The activation arrays are not touched. Run once per store.
"""
import os, pathlib, shutil
import numpy as np
import hydra
from hydra.utils import get_original_cwd
import midi._compat            # noqa: F401  -- torch.load default
from midi import qm9_labels
from midi.datasets import qm9_dataset


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg):
    # Hydra runs with chdir: True, so the process cwd is the run directory
    # under outputs/, NOT the directory you typed the command in. A relative
    # +store therefore resolves against that run directory, which is how this
    # first failed looking for
    #   .../outputs/2026-11-11/13-26-21-h_ada_sampling/activations_test_full
    # Resolve against the ORIGINAL cwd instead.
    store = cfg.store
    if not os.path.isabs(store):
        store = os.path.join(get_original_cwd(), store)
    store = os.path.abspath(store)
    path = os.path.join(store, "labels.npz")
    if not os.path.exists(path):
        raise SystemExit(f"no labels.npz at {path}\n"
                         f"(+store={cfg.store!r} resolved against the original "
                         f"cwd {get_original_cwd()!r})")

    # Second trap, worse than the first. QM9DataModule resolves the dataset
    # root as Path(get_original_cwd()).parents[0] / cfg.dataset.datadir, so
    # this must be run from <repo>/midi. From the repo root, parents[0] is the
    # repo's PARENT, and the dataset is looked for -- or re-downloaded -- there.
    data_root = pathlib.Path(get_original_cwd()).parents[0] / cfg.dataset.datadir
    if not data_root.exists():
        raise SystemExit(
            f"dataset root {data_root} does not exist.\n"
            f"Run this from <repo>/midi, not from {get_original_cwd()}.\n"
            f"The data root is derived from the parent of the working "
            f"directory, so the working directory decides where the dataset "
            f"is looked for.")

    old = dict(np.load(path, allow_pickle=False))
    print(f"[backfill] {path}: {len(old)} existing keys, "
          f"{len(old['dataset_pos']):,} molecules")

    dm = qm9_dataset.QM9DataModule(cfg)
    di = qm9_dataset.QM9infos(datamodule=dm, cfg=cfg)
    ds = {"train": dm.train_dataloader(), "val": dm.val_dataloader(),
          "test": dm.test_dataloader()}[cfg.split].dataset

    mol_ids = old["dataset_pos"].tolist()
    if max(mol_ids) >= len(ds):
        raise SystemExit(f"store references position {max(mol_ids)} but the "
                         f"{cfg.split} split has only {len(ds)} molecules")

    comp = qm9_labels.composition_counts(ds, mol_ids, di.atom_decoder)
    print("[backfill] verify:",
          qm9_labels.verify_composition(comp, old["n_atoms"]))

    # Keep a backup, then write atomically: a half-written labels.npz would be
    # worse than no change at all.
    shutil.copy2(path, path + ".bak")
    merged = dict(old); merged.update(comp)
    # Write through an open file handle, NOT a filename. np.savez appends
    # ".npz" to any string target that does not already end in it, so
    # np.savez("labels.npz.tmp", ...) silently creates "labels.npz.tmp.npz"
    # and the os.replace below would then fail on a path that does not exist.
    # A file object is written exactly where it points.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **merged)
    os.replace(tmp, path)
    print(f"[backfill] wrote {len(merged)} keys; backup at {path}.bak")

    for k in sorted(comp):
        v = comp[k]
        print(f"  {k:22s} min={v.min():3d} max={v.max():3d} mean={v.mean():6.2f}")


if __name__ == "__main__":
    main()