"""scripts/check_novelty.py -- is novelty comparing like with like?

    python3.11 -u scripts/check_novelty.py [<arm_dir>]

Novelty in MiDi is a plain string membership test:

    for smiles in unique:
        if smiles not in self.train_smiles:
            novel.append(smiles)

so it is only meaningful if both sides of that test write the same molecule
the same way. They are not built the same way, and this script measures how
much that matters.

The TRAINING side, midi/datasets/qm9_dataset.py lines 144-156:

    suppl  = Chem.SDMolSupplier(raw, removeHs=False, sanitize=False)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

so: hydrogens explicit, molecule NOT sanitised, stereochemistry dropped.

The GENERATED side, midi/metrics/molecular_metrics.py compute_validity:

    Chem.SanitizeMol(largest_mol)
    smiles = Chem.MolToSmiles(largest_mol)

so: hydrogens explicit (they are graph nodes), molecule SANITISED,
isomericSmiles at its default of True.

Two mismatches follow. Sanitisation performs aromaticity perception, so an
unsanitised molecule can be written in Kekule form where a sanitised one is
written with lowercase aromatic atoms, and those are different strings for
the same molecule. And isomericSmiles differs, so a stereocentre is written
on one side and not the other.

Either mismatch makes a training molecule look novel when it is not, which
pushes novelty UP. The open discrepancy runs the other way (62.95 measured
against 67.5 published), so a mismatch is not on its own the explanation,
but it has to be quantified before novelty can be an admissibility gate.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from rdkit import Chem, RDLogger

from midi import metrics_eval as me

RDLogger.DisableLog("rdApp.*")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PICKLE = os.path.join(REPO, "data", "qm9", "qm9_pyg", "processed",
                            "train_smiles.pickle")


def describe(name, smis, k=4):
    smis = list(smis)
    n = len(smis)
    with_h = sum(1 for s in smis if "[H]" in s)
    lower = sum(1 for s in smis if any(c in s for c in "cnops"))
    print(f"{name}: {n:,} strings")
    print(f"  contain explicit [H]      {100.0 * with_h / max(n, 1):.1f}%")
    print(f"  contain lowercase aromatic {100.0 * lower / max(n, 1):.1f}%")
    for s in smis[:k]:
        print(f"    {s}")
    print()


def recanon(s, remove_h=False, isomeric=True):
    """Re-parse and rewrite a SMILES, optionally dropping explicit H."""
    m = Chem.MolFromSmiles(s, sanitize=True)
    if m is None:
        return None
    if remove_h:
        m = Chem.RemoveHs(m)
    return Chem.MolToSmiles(m, isomericSmiles=isomeric)


def novelty(gen_unique, train, label):
    hits = sum(1 for s in gen_unique if s in train)
    n = len(gen_unique)
    nov = 1.0 - hits / n if n else float("nan")
    print(f"  {label:<46} matched {hits:>5} / {n:<5}  novelty {100 * nov:6.2f}%")
    return nov


def main(arm_dir=None):
    if not os.path.exists(TRAIN_PICKLE):
        raise SystemExit(f"training SMILES not found at {TRAIN_PICKLE}")
    with open(TRAIN_PICKLE, "rb") as f:
        train = pickle.load(f)
    train = set(train)
    describe("TRAIN (as stored, what MiDi compares against)", sorted(train))

    arm_dir = arm_dir or os.path.join(REPO, "results", "sweep", "control")
    from analyse_pilot import load_arm
    recs = list(load_arm(arm_dir).values())
    out = me.set_metrics(recs)
    gen_unique = out["unique_smiles"]
    print(f"generated set: {arm_dir}")
    describe("GENERATED (as produced)", gen_unique)

    print("NOVELTY under each way of comparing the two sides")
    print("  (lower novelty means MORE matches with the training partition)")
    novelty(gen_unique, train, "as-is, which is what MiDi actually does")

    # Both sides through the same round trip.
    tr_canon = {c for c in (recanon(s) for s in train) if c}
    gen_canon = {c for c in (recanon(s) for s in gen_unique) if c}
    novelty(gen_canon, tr_canon, "both re-canonicalised, explicit H kept")

    tr_noh = {c for c in (recanon(s, remove_h=True) for s in train) if c}
    gen_noh = {c for c in (recanon(s, remove_h=True) for s in gen_unique) if c}
    novelty(gen_noh, tr_noh, "both re-canonicalised, hydrogens removed")

    tr_ni = {c for c in (recanon(s, isomeric=False) for s in train) if c}
    gen_ni = {c for c in (recanon(s, isomeric=False) for s in gen_unique) if c}
    novelty(gen_ni, tr_ni, "both re-canonicalised, isomericSmiles=False")

    tr_both = {c for c in (recanon(s, True, False) for s in train) if c}
    gen_both = {c for c in (recanon(s, True, False) for s in gen_unique) if c}
    novelty(gen_both, tr_both, "no H and no stereo, the loosest match")

    print()
    print(f"re-canonicalisation failures: train "
          f"{len(train) - len(tr_canon):,} of {len(train):,}, "
          f"generated {len(gen_unique) - len(gen_canon)} of {len(gen_unique)}")
    print()
    print("VERDICT")
    print("  If the as-is figure is far from the re-canonicalised ones, the")
    print("  two sides are NOT writing molecules the same way and novelty is")
    print("  measuring string formatting as much as chemistry. In that case")
    print("  the reported metric should be the matched-canonicalisation one,")
    print("  and the manuscript has to say which was used, because the")
    print("  published 67.5% was computed with MiDi's as-is comparison.")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
