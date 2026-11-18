"""midi/metrics_eval.py -- the degradation metrics for any generated set.

Stage 1 of results/tolerance_protocol.md (commits bf3dec5, 57e911b). Reads
the stored graph records straight out of the chunk_*.pt files, so every set
already on disk is evaluable and nothing is regenerated.

Definitions come from MiDi, by calling MiDi, not by reimplementation. The one
exception is the float-corrected stability variant, which cannot reuse
check_stability because Molecule.__init__ forces an integer bond tensor; that
variant mirrors the same arithmetic and a test requires the two to agree
exactly on every molecule with no aromatic bond.

THE ARITHMETIC, in one place.

Let the set hold N generated molecules, molecule i having a_i atoms.

1. A PROPORTION. Validity, connectivity and molecule stability are all
   k out of N:  p = k / N. Nothing more.

2. ATOM STABILITY IS A RATIO OF SUMS, NOT A MEAN OF RATIOS. Write s_i for the
   number of stable atoms in molecule i. The two candidates are

       mean of ratios   (1/N) * sum_i (s_i / a_i)
       ratio of sums    (sum_i s_i) / (sum_i a_i)

   and they are different numbers whenever the a_i differ. MiDi takes the
   second, and the line that does it is

       self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

   because a weighted mean with weight equal to the denominator collapses:

       sum_i (w_i * v_i) / sum_i w_i          with v_i = s_i/a_i, w_i = a_i
     = sum_i (a_i * s_i / a_i) / sum_i a_i
     = sum_i s_i / sum_i a_i

   so the weight is what makes the metric about ATOMS rather than about
   molecules. (`num_bonds` is MiDi's name for what it returns; the value is
   len(atom_types), the atom count.)

3. BINOMIAL STANDARD ERROR. A proportion p measured on n independent items
   has sampling standard deviation

       sigma_bin = sqrt( p * (1 - p) / n )

   p is the observed proportion, n the number of items it was measured over.
   At p = 0.98 and n = 500 this is sqrt(0.98*0.02/500) = 0.00626, that is
   0.63 percentage points. This is reported per metric so the tolerance rule
   in section 4 of the protocol can take sigma_eff = max(s, sigma_bin).

   It applies cleanly to validity, connectivity and molecule stability, whose
   denominator is N and fixed by design. It does NOT apply cleanly to
   uniqueness and novelty, whose denominators are themselves random (the
   valid count, then the unique count), nor to atom stability, whose items
   are atoms that are correlated within a molecule. For those the number is
   reported as an indication and the matched-n subsampling of Stage 2 is what
   the tolerance actually rests on.

4. VALENCY, and the aromatic truncation. check_stability sums the bond codes
   along each row of the bond matrix,

       valency_j = sum_k B[j, k]

   after mapping code 4 (aromatic) to 1.5 and negatives to 0. But B is
   int64, fixed by an assertion in Molecule.__init__, so 1.5 stores as 1 and
   an aromatic ring atom gets valency 2 instead of 3. Since 2 is not in the
   allowed set [3, 4] for neutral carbon, every aromatic ring atom is counted
   unstable. That is MiDi's real behaviour and stays primary; the float
   variant is reported beside it with the aromatic-bond rate, which is the
   whole difference between them.
"""
import hashlib
import os
import platform
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem, RDLogger

from midi.analysis.rdkit_functions import Molecule, allowed_bonds, check_stability

RDLogger.DisableLog("rdApp.*")
ATOM_DECODER = ("H", "C", "N", "O", "F")


# ------------------------------------------------------------------ helpers

def binom_se(k, n):
    """sqrt(p(1-p)/n) with p = k/n. nan when n is 0."""
    if n <= 0:
        return float("nan")
    p = k / n
    return float(np.sqrt(p * (1.0 - p) / n))


def proportion(k, n):
    """(p, k, n, sigma_bin) for a count k out of n."""
    return {"p": (k / n if n else float("nan")), "k": int(k), "n": int(n),
            "se_binom": binom_se(k, n)}


def to_molecule(rec, atom_decoder=ATOM_DECODER):
    """A MiDi Molecule from one stored record, or None if it will not build.

    Molecule.__init__ asserts long dtypes and builds self.rdkit_mol at
    construction time, BEFORE anything mutates bond_types. That ordering is
    what keeps the rdkit mol clean, and it is why stability must be given a
    clone rather than this object.
    """
    try:
        return Molecule(
            atom_types=torch.as_tensor(rec["atom_types"]).long(),
            bond_types=torch.as_tensor(rec["bond_types"]).long(),
            positions=torch.as_tensor(rec["positions"]).float(),
            charges=torch.as_tensor(rec["charges"]).long(),
            atom_decoder=list(atom_decoder))
    except Exception:
        return None


# ------------------------------------------------------------- the metrics

def validity_midi(mol):
    """MiDi's convention: largest fragment, sanitise that, return its SMILES.

    Mirrors compute_validity in midi/metrics/molecular_metrics.py. A
    disconnected molecule is VALID when its largest piece sanitises, and the
    SMILES that feeds uniqueness and novelty is that piece alone.

    Returns (smiles or None, n_fragments).
    """
    raw = getattr(mol, "rdkit_mol", None)
    if raw is None:
        return None, 0
    try:
        frags = Chem.rdmolops.GetMolFrags(raw, asMols=True, sanitizeFrags=False)
    except Exception:
        return None, 0
    n_frag = len(frags)
    if n_frag == 0:
        return None, 0
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    try:
        probe = Chem.Mol(largest)          # copy: SanitizeMol mutates
        Chem.SanitizeMol(probe)
        smi = Chem.MolToSmiles(probe)
        return (smi if smi else None), n_frag
    except Exception:
        return None, n_frag


def validity_strict(mol):
    """Whole molecule must sanitise. This is _count_valid in generate.py, and
    it is what the valid_pct column of every summary.csv holds. Secondary,
    never mixed with the MiDi figure in one table."""
    raw = getattr(mol, "rdkit_mol", None)
    if raw is None:
        return None
    try:
        probe = Chem.Mol(raw)
        Chem.SanitizeMol(probe)
        smi = Chem.MolToSmiles(probe)
        return smi if smi else None
    except Exception:
        return None


def stability_int(mol):
    """MiDi's check_stability, on a CLONE because it mutates its argument.

    Returns (mol_stable 0/1, n_stable_atoms, n_atoms).
    """
    clone = type("M", (), {})()
    clone.atom_types = mol.atom_types.clone()
    clone.bond_types = mol.bond_types.clone()
    clone.charges = mol.charges.clone()
    ms, ns, na = check_stability(clone, None, atom_decoder=list(ATOM_DECODER))
    return int(ms.item()), int(ns.item()), int(na)


def stability_float(mol, atom_decoder=ATOM_DECODER):
    """The same arithmetic with a FLOAT bond tensor, so aromatic counts 1.5.

    Not MiDi's definition. Reported beside it because the gap between the two
    is exactly the aromatic-bond rate, and a steered arm that changes how
    often rings form would move the primary metric for an arithmetic reason
    rather than a chemical one. A test requires the two to agree exactly on
    molecules with no aromatic bond.
    """
    e = mol.bond_types.clone().float()
    e[e == 4] = 1.5
    e[e < 0] = 0.0
    val = e.sum(dim=-1)
    n_stable, mol_stable = 0, True
    for t, v, c in zip(mol.atom_types.tolist(), val.tolist(),
                       mol.charges.tolist()):
        possible = allowed_bonds[atom_decoder[int(t)]]
        if isinstance(possible, int):
            ok = possible == v
        elif isinstance(possible, dict):
            exp = possible[c] if c in possible else possible[0]
            ok = (exp == v) if isinstance(exp, int) else (v in exp)
        else:
            ok = v in possible
        n_stable += int(ok)
        if not ok:
            mol_stable = False
    return int(mol_stable), n_stable, len(mol.atom_types)


def aromatic_counts(mol):
    """(n_aromatic_bonds, n_bonds, has_any_aromatic) from the STORED codes.

    MEASURED 2026-11-15 on the sweep's control arm: the stored bond codes are
    {0: 155747, 1: 8656, 2: 476, 3: 148}. **Code 4 never appears.** The
    generator emits single, double and triple bonds only.

    Two consequences, and they go in opposite directions.

    The aromatic truncation in check_stability is INERT on this data. Its
    `edge_types[edge_types == 4] = 1.5` has nothing to match, so the integer
    and float-corrected stability variants are identical. Confirmed on the
    ten baselines: both give 0.97220 molecule stability and 0.99798 atom
    stability, to five decimals. The behaviour is still real and still worth
    documenting for any dataset whose generator does emit code 4, but it
    cannot move the primary metric here, and the earlier worry that it might
    was wrong.

    The NOVELTY mechanism is unaffected. Aromaticity in a generated SMILES
    comes from RDKit PERCEIVING an aromatic ring while sanitising alternating
    single and double bonds, not from a stored code 4. That is why 14.5% of
    generated SMILES carry lowercase aromatic atoms while the training side,
    read with sanitize=False, carries none. The quantity that drives the
    novelty bias is therefore the rate in the WRITTEN SMILES, not here. See
    `aromatic_smiles` below.
    """
    B = mol.bond_types.numpy()
    U = np.triu(B, 1)
    n_bonds = int((U > 0).sum())
    n_arom = int((U == 4).sum())
    return n_arom, n_bonds, int(n_arom > 0)


def aromatic_smiles(smi):
    """Does this canonical SMILES carry a perceived aromatic atom?

    This, not the stored bond code, is the quantity that must sit beside
    novelty. The training partition is Kekule throughout because it was read
    with sanitize=False, so every generated molecule written with lowercase
    aromatic atoms fails the string membership test even when the molecule is
    in the training set. The bias in novelty is proportional to this rate, so
    an arm whose value differs from baseline moves its novelty for a reason
    that is string formatting rather than chemistry.

    Lowercase organic-subset aromatic atoms are b, c, n, o, p, s. Bracketed
    atoms such as [nH] are covered by the same letters.
    """
    if not smi:
        return 0
    return int(any(ch in smi for ch in "bcnops"))


# ------------------------------------------------------------------- driver

def set_metrics(recs, train_smiles=None, atom_decoder=ATOM_DECODER):
    """Every full-n metric for one generated set.

    recs: list of dicts with atom_types, charges, bond_types, positions.
    train_smiles: iterable of training SMILES for novelty, or None to skip.
                  Passed as a set; membership is plain string equality on the
                  canonical form, exactly as MiDi does it.

    The matched-n metrics (uniqueness, novelty, FCD) are returned here at the
    set's own n for reference ONLY. Admissibility uses the Stage 2 harness at
    m = 250, because their denominators move with validity.
    """
    N = len(recs)
    mols = [to_molecule(r, atom_decoder) for r in recs]

    smiles_midi, n_frags = [], []
    n_strict = 0
    s_sum = a_sum = ms_sum = 0
    sf_sum = af_sum = msf_sum = 0
    ar_bonds = tot_bonds = ar_mols = 0

    for m in mols:
        if m is None:
            n_frags.append(0)
            continue
        smi, nf = validity_midi(m)
        n_frags.append(nf)
        if smi is not None:
            smiles_midi.append(smi)
        if validity_strict(m) is not None:
            n_strict += 1

        ms, ns, na = stability_int(m)
        ms_sum += ms; s_sum += ns; a_sum += na
        msf, nsf, naf = stability_float(m, atom_decoder)
        msf_sum += msf; sf_sum += nsf; af_sum += naf

        nb, tb, hv = aromatic_counts(m)
        ar_bonds += nb; tot_bonds += tb; ar_mols += hv

    n_valid = len(smiles_midi)
    uniq = sorted(set(smiles_midi))
    out = {
        "n_generated": N,
        "validity_midi": proportion(n_valid, N),
        "validity_strict": proportion(n_strict, N),
        "connectivity": proportion(sum(1 for f in n_frags if f == 1), N),
        "mol_stable": proportion(ms_sum, N),
        # Ratio of sums, not mean of ratios. See the module docstring.
        "atom_stable": proportion(s_sum, a_sum),
        "mol_stable_float": proportion(msf_sum, N),
        "atom_stable_float": proportion(sf_sum, af_sum),
        # Stored bond code 4. Measured identically zero on this generator,
        # which is the finding that the stability truncation is inert here.
        "aromatic_bond_rate": proportion(ar_bonds, tot_bonds),
        "aromatic_mol_rate": proportion(ar_mols, N),
        # PERCEIVED aromaticity in the written SMILES. THIS is the rate that
        # drives the novelty bias and must be read beside it.
        "aromatic_smiles_rate": proportion(
            sum(aromatic_smiles(s) for s in smiles_midi), len(smiles_midi)),
        "frag_mean": float(np.mean(n_frags)) if n_frags else float("nan"),
        "frag_max": int(np.max(n_frags)) if n_frags else 0,
        "uniqueness_own_n": proportion(len(uniq), n_valid),
        "smiles": smiles_midi,
        "unique_smiles": uniq,
    }
    if train_smiles is not None:
        ts = set(train_smiles)
        novel = [s for s in uniq if s not in ts]
        out["novelty_own_n"] = proportion(len(novel), len(uniq))
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _dist_version(name):
    """A package's version from its installed metadata.

    `getattr(mod, "__version__", ...)` is not enough: fcd_torch exposes no
    __version__ at all and reported "unknown", which pins nothing. The
    distribution metadata is what pip actually installed.
    """
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "not installed"


def versions(checkpoint=None):
    """Step 4 of the protocol: pinned by the module, not by hand.

    Everything a reported number depends on. Two entries beyond the obvious:

    - `chemnet_sha256`. FCD is only reproducible against the exact ChemNet
      weights, and fcd_torch ships them inside the package as
      ChemNet_v0.13_pretrained.pt. A version string alone would not catch a
      package that silently reshipped different weights.
    - `git_commit`. The metric DEFINITIONS live in this repository and three
      of them were corrected on 2026-11-15, so a number is only interpretable
      alongside the commit that produced it.
    """
    import rdkit
    v = {"python": platform.python_version(),
         "numpy": np.__version__,
         "torch": torch.__version__,
         "rdkit": rdkit.__version__,
         "scipy": _dist_version("scipy"),
         "fcd_torch": _dist_version("fcd_torch")}

    try:
        import fcd_torch
        w = os.path.join(os.path.dirname(fcd_torch.__file__),
                         "ChemNet_v0.13_pretrained.pt")
        if os.path.exists(w):
            v["chemnet_weights"] = os.path.basename(w)
            v["chemnet_sha256"] = _sha256(w)
    except Exception:
        pass

    try:
        import subprocess
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
            text=True, timeout=20).stdout.strip() or "unknown"
        v["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo, capture_output=True, text=True,
            timeout=20).stdout.strip())
    except Exception:
        v["git_commit"] = "unknown"

    if checkpoint and os.path.exists(checkpoint):
        v["checkpoint"] = os.path.basename(checkpoint)
        v["checkpoint_sha256"] = _sha256(checkpoint)
    return v
