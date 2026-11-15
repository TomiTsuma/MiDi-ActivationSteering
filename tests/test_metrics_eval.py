"""tests/test_metrics_eval.py -- the degradation metrics, on a toy set.

One continuous example runs through every test here: four molecules, 18 atoms
between them, chosen so each metric comes out as a number you can check by
hand.

    water    O + 2H        3 atoms   all stable
    methane  C + 4H        5 atoms   all stable
    ring     6 aromatic C  6 atoms   0 stable under MiDi's integer valency
    ammonia  N + 3H        4 atoms   all stable

                            atoms  3 + 5 + 6 + 4 = 18
    MiDi (integer bonds)   stable  3 + 5 + 0 + 4 = 12
    float-corrected        stable  3 + 5 + 6 + 4 = 18

so atom stability is 12/18 = 0.6667 primary and 18/18 = 1.0 corrected, and
the whole gap is the aromatic ring. Molecule stability is 3/4 = 0.75 primary.

The mean of the per-molecule ratios is (1 + 1 + 0 + 1)/4 = 0.75, which is NOT
0.6667. That difference is the subject of test_atom_stability_is_ratio_of_sums
and it is the reason MiDi passes weight=num_bonds.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi import metrics_eval as me

slow = pytest.mark.slow
AD = me.ATOM_DECODER                       # ("H", "C", "N", "O", "F")


def rec(symbols, bonds, charges=None):
    """A stored-record dict from element symbols and (i, j, bond_code)."""
    n = len(symbols)
    at = np.array([AD.index(s) for s in symbols], dtype=np.int64)
    bt = np.zeros((n, n), dtype=np.int64)
    for i, j, c in bonds:
        bt[i, j] = c
        bt[j, i] = c
    pos = np.arange(3 * n, dtype=np.float32).reshape(n, 3) * 0.5
    ch = np.zeros(n, dtype=np.int64) if charges is None else np.asarray(charges)
    return {"atom_types": at, "bond_types": bt, "positions": pos,
            "charges": ch.astype(np.int64)}


WATER = rec(["O", "H", "H"], [(0, 1, 1), (0, 2, 1)])
METHANE = rec(["C", "H", "H", "H", "H"],
              [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1)])
RING = rec(["C"] * 6, [(i, (i + 1) % 6, 4) for i in range(6)])
AMMONIA = rec(["N", "H", "H", "H"], [(0, 1, 1), (0, 2, 1), (0, 3, 1)])
TOY = [WATER, METHANE, RING, AMMONIA]


# ------------------------------------------------------- the arithmetic

def test_binom_se():
    """sigma = sqrt(p(1-p)/n), the one formula the tolerance rule needs."""
    assert abs(me.binom_se(490, 500) - np.sqrt(0.98 * 0.02 / 500)) < 1e-15
    assert abs(me.binom_se(490, 500) - 0.0062610) < 1e-6   # 0.63 pp
    assert abs(me.binom_se(3, 4) - np.sqrt(0.75 * 0.25 / 4)) < 1e-15
    assert me.binom_se(500, 500) == 0.0        # p = 1 has no spread
    assert np.isnan(me.binom_se(0, 0))


def test_proportion_bookkeeping():
    p = me.proportion(3, 4)
    assert p["k"] == 3 and p["n"] == 4
    assert abs(p["p"] - 0.75) < 1e-15
    assert abs(p["se_binom"] - 0.21650635) < 1e-7


def test_stability_of_each_toy_molecule():
    """The per-molecule numbers the whole example rests on."""
    expect = {"water": (WATER, 1, 3, 3), "methane": (METHANE, 1, 5, 5),
              "ring": (RING, 0, 0, 6), "ammonia": (AMMONIA, 1, 4, 4)}
    for name, (r, ms, ns, na) in expect.items():
        m = me.to_molecule(r)
        assert m is not None, name
        got = me.stability_int(m)
        assert got == (ms, ns, na), f"{name}: {got} != {(ms, ns, na)}"


def test_atom_stability_is_ratio_of_sums_not_mean_of_ratios():
    """The subtlest step, as two different numbers.

        ratio of sums   sum(s_i) / sum(a_i) = 12/18 = 0.6667
        mean of ratios  (1/N) sum(s_i/a_i)  = 0.75

    MiDi takes the first, via weight=num_bonds in the MeanMetric update, and
    the identity that makes the weight do it is

        sum(w_i v_i) / sum(w_i)  with v_i = s_i/a_i, w_i = a_i
      = sum(a_i * s_i/a_i) / sum(a_i) = sum(s_i)/sum(a_i)
    """
    out = me.set_metrics(TOY)
    per = [me.stability_int(me.to_molecule(r)) for r in TOY]
    s = sum(p[1] for p in per)
    a = sum(p[2] for p in per)
    assert (s, a) == (12, 18)

    ratio_of_sums = s / a
    mean_of_ratios = float(np.mean([p[1] / p[2] for p in per]))
    assert abs(ratio_of_sums - 12 / 18) < 1e-15
    assert abs(mean_of_ratios - 0.75) < 1e-15
    assert abs(ratio_of_sums - mean_of_ratios) > 0.08     # they differ

    # The module reports the ratio of sums, as MiDi does.
    assert abs(out["atom_stable"]["p"] - 12 / 18) < 1e-15
    assert out["atom_stable"]["n"] == 18

    # And the weighted-mean identity, arrived at numerically.
    w = np.array([p[2] for p in per], float)
    v = np.array([p[1] / p[2] for p in per], float)
    assert abs(float((w * v).sum() / w.sum()) - ratio_of_sums) < 1e-15


def test_molecule_stability_is_a_plain_proportion():
    out = me.set_metrics(TOY)
    assert out["mol_stable"]["k"] == 3 and out["mol_stable"]["n"] == 4
    assert abs(out["mol_stable"]["p"] - 0.75) < 1e-15


# --------------------------------------------- the aromatic truncation

def test_float_variant_agrees_on_every_non_aromatic_molecule():
    """The gate on the corrected variant: it must change nothing except
    aromatic molecules, or it is not the same arithmetic."""
    for r in (WATER, METHANE, AMMONIA):
        m = me.to_molecule(r)
        assert me.stability_int(m) == me.stability_float(m)


def test_aromatic_ring_is_the_whole_difference():
    """Integer: valency 2 per carbon, 2 not in [3, 4], 0 of 6 stable.
    Float:   valency 3 per carbon, 3 in [3, 4], 6 of 6 stable."""
    m = me.to_molecule(RING)
    assert me.stability_int(m) == (0, 0, 6)
    assert me.stability_float(m) == (1, 6, 6)

    out = me.set_metrics(TOY)
    assert abs(out["atom_stable"]["p"] - 12 / 18) < 1e-15
    assert abs(out["atom_stable_float"]["p"] - 1.0) < 1e-15
    assert abs(out["mol_stable"]["p"] - 0.75) < 1e-15
    assert abs(out["mol_stable_float"]["p"] - 1.0) < 1e-15
    # 6 of the 9 bonds in the toy set are aromatic (3 + 4 + 6 + 3 = 16? no:
    # water 2, methane 4, ring 6, ammonia 3 = 15 bonds, 6 of them aromatic)
    assert out["aromatic_bond_rate"]["n"] == 15
    assert out["aromatic_bond_rate"]["k"] == 6
    assert out["aromatic_mol_rate"]["k"] == 1


def test_check_stability_input_is_not_mutated():
    """check_stability writes into its argument's bond_types. The module must
    hand it a clone, or everything computed afterwards is corrupted."""
    m = me.to_molecule(RING)
    before = m.bond_types.clone()
    me.stability_int(m)
    assert (m.bond_types == before).all(), \
        "stability_int let check_stability mutate the molecule"


# ----------------------------------------------- the two validity rules

def test_midi_validity_takes_the_largest_fragment():
    """A set with two fragments where the SMALLER one is chemically impossible.

    MiDi keeps the largest fragment and sanitises only that, so the molecule
    counts as valid. The strict rule sanitises the whole graph and rejects it.
    That is the entire difference between the two columns, and it is why they
    must never share a table.
    """
    # methane (5 atoms, fine) plus a neutral O carrying three H (impossible)
    sym = ["C", "H", "H", "H", "H", "O", "H", "H", "H"]
    bonds = [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1),
             (5, 6, 1), (5, 7, 1), (5, 8, 1)]
    m = me.to_molecule(rec(sym, bonds))
    assert m is not None

    smi, n_frag = me.validity_midi(m)
    assert n_frag == 2, f"expected 2 fragments, got {n_frag}"
    # The SMILES keeps its hydrogens EXPLICIT, because this is the
    # explicit-H setting and MiDi builds every atom from the graph rather
    # than letting RDKit infer H. Methane comes back as
    # "[H]C([H])([H])[H]", not "C". See
    # test_canonical_smiles_keep_explicit_hydrogens for why that matters.
    assert smi == "[H]C([H])([H])[H]", \
        f"largest fragment should be methane, got {smi!r}"
    assert me.validity_strict(m) is None, \
        "a neutral O with three H should fail whole-molecule sanitisation"


def test_canonical_smiles_keep_explicit_hydrogens():
    """Load-bearing for novelty, so it is pinned here.

    Every generated SMILES carries its hydrogens explicitly. Novelty is a
    plain string membership test against the checkpoint's training SMILES, so
    if those are stored in the usual implicit-H form ("C" for methane) then
    NOTHING ever matches and novelty reads as 100% for any set.

    The lever, measured here: a round trip through Chem.CanonSmiles COLLAPSES
    the explicit hydrogens, so "[H]C([H])([H])[H]" becomes "C". Whether
    novelty is meaningful therefore depends entirely on whether both sides
    went through the same transformation, and nothing in MiDi's
    `smiles not in self.train_smiles` guarantees that they did.

    Confirming the stored form on both sides is Stage 3, and it is the first
    candidate for the open 62.95 against 67.5 gap. Until it is confirmed,
    novelty is not fit to be an admissibility gate.
    """
    from rdkit import Chem
    smi, _ = me.validity_midi(me.to_molecule(METHANE))
    assert smi == "[H]C([H])([H])[H]"            # as generated: explicit H
    assert smi != Chem.CanonSmiles("C")          # so a raw compare fails
    assert Chem.CanonSmiles(smi) == "C"          # but a round trip bridges it
    water, _ = me.validity_midi(me.to_molecule(WATER))
    assert water == "[H]O[H]", water
    assert Chem.CanonSmiles(water) == "O"


def test_connectivity_is_separate_from_validity():
    sym = ["C", "H", "H", "H", "H", "O", "H", "H"]
    bonds = [(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1),
             (5, 6, 1), (5, 7, 1)]
    out = me.set_metrics([rec(sym, bonds)] + TOY)
    assert out["connectivity"]["k"] == 4 and out["connectivity"]["n"] == 5
    assert out["frag_max"] == 2


def test_uniqueness_and_novelty_on_the_toy():
    """Two copies of water among the four: 4 valid SMILES, 3 distinct."""
    out = me.set_metrics(TOY + [WATER], train_smiles={"O"})
    u = out["uniqueness_own_n"]
    assert u["n"] == out["validity_midi"]["k"]
    assert u["k"] == len(set(out["smiles"]))
    # water appears twice, so distinct < valid
    assert u["k"] < u["n"]
    nv = out["novelty_own_n"]
    assert nv["n"] == u["k"]
    assert "O" not in [s for s in out["unique_smiles"]
                       if s in {"O"}] or nv["k"] == u["k"] - 1


def test_versions_records_what_the_protocol_pins():
    v = me.versions()
    for k in ("python", "numpy", "torch", "rdkit", "fcd_torch"):
        assert k in v and v[k]


@slow
def test_strict_validity_matches_count_valid_on_the_real_control():
    """The assertion the protocol promises: the module's strict figure must
    equal the valid_pct column that _count_valid wrote into summary.csv."""
    import csv
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep = os.path.join(repo, "results", "sweep")
    if not os.path.isdir(os.path.join(sweep, "control")):
        pytest.skip("results/sweep/control not present")
    sys.path.insert(0, os.path.join(repo, "scripts"))
    from analyse_pilot import load_arm

    recs = list(load_arm(os.path.join(sweep, "control")).values())
    out = me.set_metrics(recs)
    with open(os.path.join(sweep, "summary.csv"), newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["arm"] == "control")
    assert abs(100.0 * out["validity_strict"]["p"]
               - float(row["valid_pct"])) < 0.21, \
        (out["validity_strict"], row["valid_pct"])
