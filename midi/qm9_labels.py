"""QM9 DFT labels, aligned to the processed dataset's molecule order.

Why this module is needed
-------------------------
MiDi is an unconditional model, so `QM9Dataset.process()` reads the DFT label
CSV only as an index filter and never attaches the labels to the stored
molecules. The processed `.pt` therefore has no polarisability, no heat
capacity, no HOMO, no LUMO and no dipole moment. They live only in
`data/qm9/qm9_pyg/raw/`.

The alignment has one trap. `download()` writes each split CSV from a SHUFFLED
frame (`df.sample(frac=1, random_state=42)`), but `process()` iterates the SDF
in increasing order and appends survivors in that order:

    for i, mol in enumerate(suppl):
        if i in skip or i not in target_df.index:
            continue
        data_list.append(data)

So dataset position j corresponds to the j-th SMALLEST surviving SDF index, not
to CSV row j. Joining on CSV row order silently mislabels every molecule, and
the symptom is a flat probe that looks like a negative scientific result.

Verified against the real files for the test split: 13,388 CSV rows, 333
dropped as uncharacterized, 13,055 survivors; CSV index i has mol_id
gdb_{i+1}, so index i is SDF molecule i (0-based).

Notation
--------
    sdf_index   0-based position of a molecule in gdb9.sdf (and in gdb9.sdf.csv)
    dataset_pos 0-based position in the processed split, what the sampler indexes
    skip        the 3,054 molecules listed in uncharacterized.txt
"""

import csv
import json
import os

import numpy as np

# The five QM9 DFT labels this study targets, as CSV column -> our name.
# Units are those of the QM9 release: alpha in Bohr^3, cv in cal/(mol K),
# mu in Debye, homo/lumo in Hartree.
DFT_COLUMNS = {
    "alpha": "polarisability",
    "cv": "heat_capacity",
    "mu": "dipole_moment",
    "homo": "homo",
    "lumo": "lumo",
}
# 'gap' is available too and is homo-lumo; Table 3.1 Note 4 commits to seven
# targets without it, so it is read but flagged as not a target.
EXTRA_COLUMNS = {"gap": "homo_lumo_gap"}

SPLIT_CSV = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def _raw_dir(datadir_root):
    """datadir_root is the dataset root, e.g. <repo>/data/qm9/qm9_pyg."""
    return os.path.join(datadir_root, "raw")


def read_skip_list(raw_dir):
    """The uncharacterized molecules, as 0-based SDF indices.

    Replicates qm9_dataset.process() exactly:
        skip = [int(x.split()[0]) - 1 for x in f.read().split('\\n')[9:-2]]
    The file's own index column is 1-based, hence the -1.
    """
    path = os.path.join(raw_dir, "uncharacterized.txt")
    with open(path, "r") as f:
        lines = f.read().split("\n")
    skip = [int(x.split()[0]) - 1 for x in lines[9:-2]]
    if len(set(skip)) != len(skip):
        raise ValueError("duplicate entries in uncharacterized.txt skip list")
    return skip


def read_split_labels(raw_dir, split):
    """Returns {sdf_index: {name: float}} for one split, from its CSV.

    Also checks the index/mol_id relation, which is what fixes the off-by-one:
    CSV index i must carry mol_id gdb_{i+1}.
    """
    path = os.path.join(raw_dir, SPLIT_CSV[split])
    wanted = dict(DFT_COLUMNS)
    wanted.update(EXTRA_COLUMNS)
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        missing = [c for c in wanted if c not in header]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}; header={header}")
        col = {c: header.index(c) for c in wanted}
        molid_col = header.index("mol_id")
        for row in reader:
            i = int(row[0])
            if row[molid_col] != f"gdb_{i + 1}":
                raise ValueError(
                    f"{path}: index {i} has mol_id {row[molid_col]}, expected gdb_{i+1}. "
                    "The index/SDF correspondence this module relies on does not hold.")
            out[i] = {wanted[c]: float(row[col[c]]) for c in wanted}
    return out


def build_alignment(datadir_root, split):
    """Map dataset position -> sdf index -> labels, for one split.

    Returns a dict with:
        sdf_index   (M,) int64, sorted ascending; sdf_index[j] is dataset pos j
        labels      {name: (M,) float64}
        n_dropped   how many split molecules were uncharacterized
        n_csv       rows in the split CSV
    The caller MUST assert len(sdf_index) == len(dataset); a mismatch means the
    processed dataset was built from different raw files and no label is
    trustworthy.
    """
    raw = _raw_dir(datadir_root)
    skip = set(read_skip_list(raw))
    per_sdf = read_split_labels(raw, split)

    surviving = sorted(i for i in per_sdf if i not in skip)
    dropped = sum(1 for i in per_sdf if i in skip)

    names = list(DFT_COLUMNS.values()) + list(EXTRA_COLUMNS.values())
    labels = {n: np.array([per_sdf[i][n] for i in surviving], dtype=np.float64)
              for n in names}
    return {
        "sdf_index": np.asarray(surviving, dtype=np.int64),
        "labels": labels,
        "n_dropped": dropped,
        "n_csv": len(per_sdf),
        "split": split,
    }


def verify_against_dataset(alignment, dataset, n_check=None, raw_dir=None, verbose=True):
    """Prove the alignment rather than assume it.

    Checks, in increasing cost:
      1. length: len(alignment) == len(dataset)
      2. atom count plausibility: QM9 with explicit H has 3..29 atoms
      3. if `raw_dir` is given, replays the SDF with the same filter and
         compares each molecule's canonical SMILES against the dataset's own
         stored `smiles` attribute. This is the only check that can catch an
         off-by-one, and it costs one pass over gdb9.sdf (about 90 seconds).

    Returns a report dict; raises on a hard mismatch.
    """
    report = {}
    sdf_index = alignment["sdf_index"]
    if len(sdf_index) != len(dataset):
        raise ValueError(
            f"alignment has {len(sdf_index)} molecules, dataset has {len(dataset)}. "
            "The processed split was not built from these raw files.")
    report["n_molecules"] = len(sdf_index)

    n_atoms = np.array([int(dataset[j].x.numel()) for j in range(len(dataset))])
    report["n_atoms_min"] = int(n_atoms.min())
    report["n_atoms_max"] = int(n_atoms.max())
    if report["n_atoms_max"] > 29 or report["n_atoms_min"] < 1:
        raise ValueError(f"atom counts out of range for QM9 with H: {report}")

    if raw_dir is not None:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        want = set(sdf_index.tolist())
        pos_of = {int(i): j for j, i in enumerate(sdf_index)}
        suppl = Chem.SDMolSupplier(os.path.join(raw_dir, "gdb9.sdf"),
                                   removeHs=False, sanitize=False)
        checked = mismatched = 0
        examples = []
        for i, mol in enumerate(suppl):
            if i not in want or mol is None:
                continue
            j = pos_of[i]
            ds_smiles = getattr(dataset[j], "smiles", None)
            if ds_smiles is None:
                break  # remove_h=True drops the smiles attribute
            sdf_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            checked += 1
            if sdf_smiles != ds_smiles:
                mismatched += 1
                if len(examples) < 5:
                    examples.append({"dataset_pos": j, "sdf_index": i,
                                     "sdf": sdf_smiles, "dataset": ds_smiles})
            if n_check is not None and checked >= n_check:
                break
        report["smiles_checked"] = checked
        report["smiles_mismatched"] = mismatched
        report["smiles_examples"] = examples
        if checked and mismatched:
            raise ValueError(
                f"{mismatched}/{checked} SMILES mismatches: the labels are NOT aligned "
                f"to the dataset order. Examples: {examples}")

    if verbose:
        print(f"  alignment verified: {report['n_molecules']} molecules, "
              f"atoms {report['n_atoms_min']}-{report['n_atoms_max']}"
              + (f", {report.get('smiles_checked', 0)} SMILES checked, "
                 f"{report.get('smiles_mismatched', 0)} mismatched"
                 if raw_dir is not None else ""))
    return report


def save(alignment, path):
    arrays = {"sdf_index": alignment["sdf_index"]}
    arrays.update(alignment["labels"])
    np.savez(path, **arrays)
    meta = {k: alignment[k] for k in ("n_dropped", "n_csv", "split")}
    with open(os.path.splitext(path)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load(path):
    z = np.load(path)
    names = list(DFT_COLUMNS.values()) + list(EXTRA_COLUMNS.values())
    return {"sdf_index": z["sdf_index"],
            "labels": {n: z[n] for n in names if n in z}}
