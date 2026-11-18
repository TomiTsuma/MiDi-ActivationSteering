"""Polarisability readout for generated molecules.

Why this module exists
----------------------
Polarisability in QM9 is a DFT observable (`alpha`, in Bohr^3). It is not a
graph property, so nothing in RDKit computes it and no amount of cheminformatics
recovers it exactly. A generated molecule has no DFT label, so the honest
options for reading its polarisability are:

  1. run a QM calculation (GFN2-xTB or DFT) on each generated molecule,
  2. predict it from the graph with a model calibrated on QM9's own labels,
  3. report a correlate and call it a correlate.

The pipeline has been doing (3) with Crippen molar refractivity (MolMR), which
is defensible -- MolMR and polarisability are linked by the Lorentz-Lorenz
relation -- but it reports the answer in the wrong units and in a quantity no
reviewer asked about. This module does (2): it fits a ridge model from graph
descriptors to the QM9 DFT polarisability label, so arm comparisons are reported
in Bohr^3 against a stated accuracy. (1) remains the gold standard and this
module is not a substitute for it; it is what can be done without a QM code.

Measured held-out accuracy on the QM9 test split (12,930 molecules that
sanitise, 70/30 molecule-level split, RidgeCV over 40 alphas):

    composition only (H,C,N,O,F counts)   R2 = 0.866   MAE = 2.14 Bohr^3
    MolMR alone                           R2 = 0.890   MAE = 1.93 Bohr^3
    composition + MolMR                   R2 = 0.930   MAE = 1.39 Bohr^3
    all 20 descriptors (this model)       R2 = 0.940   MAE = 1.28 Bohr^3

For scale, sd(alpha) = 8.13 Bohr^3 and mean(alpha) = 75.3 Bohr^3.

Reading the MAE correctly
-------------------------
MAE 1.28 Bohr^3 is the error on ONE molecule in absolute terms. It is NOT the
resolution of an arm comparison, and treating it as such would throw away a
real effect. Two reasons:

  * The arms are PAIRED. Molecule i in the dom+ arm and molecule i in the
    control start from the same size and the same noise. The surrogate is a
    deterministic function of the graph, so whatever it gets wrong about
    "molecules of this shape" is common to both members of the pair and cancels
    in the difference. What survives is the part of the error that differs
    between the two graphs.
  * The reported quantity is a MEAN over n pairs, so its standard error falls
    like 1/sqrt(n). The table computes that standard error from the paired
    differences themselves, so it already reflects both effects and needs no
    correction from the MAE.

The MAE still matters for one thing: do not report a single molecule's
polarisability to better than about +/- 1.3 Bohr^3.

Composition decomposition
-------------------------
Polarisability is strongly extensive -- a bigger molecule is a more polarisable
molecule, with no change in polarisability per atom. Composition alone already
explains R2 = 0.866 of it. So a steering arm that merely grows its molecules
will show a polarisability rise that has nothing to do with the property being
encoded, which is the same trap that `dMolMR` fell into and the reason
`analyse_sweep.py` reports a per-heavy-atom figure.

This module therefore fits TWO models on the same split:

    full     all 20 descriptors            -> alpha_hat
    comp     H,C,N,O,F counts only         -> alpha_comp

and the table reports

    d_alpha    = paired mean (alpha_hat[arm] - alpha_hat[control])
    d_comp     = paired mean (alpha_comp[arm] - alpha_comp[control])
    d_beyond   = d_alpha - d_comp

`d_beyond` is the polarisability shift that the arm's change in atomic
composition does not account for. It is the column that answers "did steering
move the property, or did it just make the molecule bigger". A large `d_alpha`
with `d_beyond` near zero is a size effect wearing the property's clothes.

Applicability domain
--------------------
The model is fitted on valid QM9 molecules. A heavily steered generation is off
that manifold by construction, and a ridge model extrapolates silently and
confidently. `predict` therefore also returns a LEVERAGE per molecule,

    h_i = z_i' (Z'Z + lam I)^-1 z_i

where z_i is the molecule's standardised descriptor row and Z the training
matrix. This is the diagonal of the fitted model's own hat matrix, so it is the
influence that row would have had in the fit, and it is the standard
applicability-domain measure in QSAR (the x-axis of a Williams plot). A molecule
is flagged when h_i exceeds the warning leverage h* = 3p/n_train.

A per-descriptor percentile box was tried first and is useless here: with 20
descriptors, "outside the 1st-99th percentile on ANY column" flags essentially
every molecule, training rows included. Leverage is a single number in the
geometry the model actually uses, and on the training rows it flags 1-2% as it
should.

Notation
--------
    alpha        QM9 DFT isotropic polarisability, Bohr^3
    alpha_hat    this model's prediction of alpha from the graph
    alpha_comp   the composition-only model's prediction of alpha
    D            (M, 20) descriptor matrix, one row per molecule
    Y            (M,) the DFT labels those rows are fitted against
    h_i          leverage of molecule i, its influence in the fitted geometry
    h*           warning leverage 3p/n_train, above which a row is flagged
"""

import json
import os

import numpy as np

# Descriptor order. Fixed, because the saved coefficients are indexed by it.
NAMES = (
    "nH", "nC", "nN", "nO", "nF",          # composition; the comp sub-model
    "MolMR", "MolLogP",                     # Crippen
    "MolWt", "NumValenceElectrons", "nHeavy",
    "bSingle", "bDouble", "bTriple", "bAromatic",
    "nRings", "nAromRings",
    "TPSA", "LabuteASA", "nRotatable", "FracCSP3",
)
COMP_COLS = (0, 1, 2, 3, 4)
ATOM_DECODER = ("H", "C", "N", "O", "F")


# ---------------------------------------------------------------- descriptors

def _bond_types():
    from rdkit import Chem
    return (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)


def rdmol_from_graph(atom_types, charges, bond_types, atom_decoder=ATOM_DECODER):
    """Build a sanitised RDKit mol from a MiDi-style graph, or None.

    atom_types  (n,)   int, indices into atom_decoder
    charges     (n,)   int, formal charges
    bond_types  (n, n) int, 0 none / 1 single / 2 double / 3 triple / 4 aromatic

    This is the ONE place a graph becomes a molecule, used for both the QM9
    training rows and the generated rows. Sharing it matters: if the training
    molecules were parsed from SMILES while the generated ones were built from
    graphs, implicit-hydrogen handling would differ and every descriptor that
    counts atoms would be measured on a different convention in the two sets.
    """
    from rdkit import Chem
    BOND = (None,) + _bond_types()
    rw = Chem.RWMol()
    for a, c in zip(np.asarray(atom_types).reshape(-1),
                    np.asarray(charges).reshape(-1)):
        if int(a) < 0:
            continue
        at = Chem.Atom(atom_decoder[int(a)])
        if int(c) != 0:
            at.SetFormalCharge(int(c))
        rw.AddAtom(at)
    E = np.triu(np.asarray(bond_types), 1)
    for i, j in zip(*np.nonzero(E)):
        b = int(E[i, j])
        if 1 <= b <= 4:
            rw.AddBond(int(i), int(j), BOND[b])
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def rdmol_from_molecule(m, atom_decoder=ATOM_DECODER):
    """Same, for a MiDi `Molecule` as `sample_batch` returns."""
    return rdmol_from_graph(m.atom_types.cpu().numpy(),
                            m.charges.cpu().numpy(),
                            m.bond_types.cpu().numpy(), atom_decoder)


def descriptors(mol):
    """The (20,) descriptor row for one sanitised RDKit mol.

    Every entry is computable from a generated graph with explicit hydrogens,
    which rules out anything needing 3-D coordinates or a conformer.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    want = _bond_types()
    bt = [0, 0, 0, 0]
    for b in mol.GetBonds():
        t = b.GetBondType()
        for i, x in enumerate(want):
            if t == x:
                bt[i] += 1
    cnt = dict.fromkeys(ATOM_DECODER, 0)
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s in cnt:
            cnt[s] += 1
    ri = mol.GetRingInfo()
    n_arom_rings = sum(
        1 for r in ri.AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    return np.array([
        cnt["H"], cnt["C"], cnt["N"], cnt["O"], cnt["F"],
        Crippen.MolMR(mol), Crippen.MolLogP(mol),
        Descriptors.MolWt(mol), Descriptors.NumValenceElectrons(mol),
        mol.GetNumHeavyAtoms(),
        bt[0], bt[1], bt[2], bt[3],
        ri.NumRings(), n_arom_rings,
        rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcLabuteASA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ], dtype=np.float64)


# ------------------------------------------------------------------- fitting

def _ridge(Dtr, Ytr, lams=None):
    """Standardise, then ridge with the lambda chosen by exact LOO.

    Returns (mu, sd, coef, intercept, lam). The intercept is unpenalised: it is
    the train mean of Y, and the centred problem is solved for the slopes. The
    same convention as probe.py, for the same reason -- penalising the intercept
    shrinks predictions toward zero rather than toward the mean, and for a
    quantity centred near 75 Bohr^3 that is a large bias.
    """
    lams = np.logspace(-3, 4, 40) if lams is None else np.asarray(lams)
    mu, sd = Dtr.mean(0), Dtr.std(0)
    sd[sd == 0] = 1.0
    Z = (Dtr - mu) / sd
    yc = Ytr - Ytr.mean()
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    Uy = U.T @ yc
    best = (np.inf, None, None)
    for lam in lams:
        f = s ** 2 / (s ** 2 + lam)                 # filter factors
        resid = yc - U @ (f * Uy)
        h = (U ** 2 * f).sum(1)                     # leverages
        press = float(((resid / np.clip(1 - h, 1e-9, None)) ** 2).mean())
        if press < best[0]:
            best = (press, lam, Vt.T @ ((s / (s ** 2 + lam)) * Uy))
    _, lam, coef = best
    return mu, sd, coef, float(Ytr.mean()), float(lam)


def _apply(mu, sd, coef, intercept, D):
    return ((np.asarray(D, float) - mu) / sd) @ coef + intercept


def fit(D, Y, seed=0, holdout=0.3):
    """Fit the full and composition-only models and measure them on a holdout.

    The split is over molecules and is reported, so the accuracy quoted in any
    write-up is a held-out number and not a training fit.
    """
    D = np.asarray(D, float)
    Y = np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y))
    ntr = int(round((1.0 - holdout) * len(Y)))
    tr, te = idx[:ntr], idx[ntr:]

    out = {"names": list(NAMES), "n_total": int(len(Y)), "n_train": int(len(tr)),
           "n_holdout": int(len(te)), "seed": int(seed),
           "y_mean": float(Y.mean()), "y_sd": float(Y.std()),
           "y_min": float(Y.min()), "y_max": float(Y.max())}

    for tag, cols in (("full", tuple(range(len(NAMES)))), ("comp", COMP_COLS)):
        mu, sd, coef, b0, lam = _ridge(D[tr][:, list(cols)], Y[tr])
        p = _apply(mu, sd, coef, b0, D[te][:, list(cols)])
        ss = ((Y[te] - Y[te].mean()) ** 2).sum()
        out[tag] = {
            "cols": list(cols), "mu": mu, "sd": sd, "coef": coef,
            "intercept": b0, "lam": lam,
            "r2": float(1.0 - ((Y[te] - p) ** 2).sum() / ss),
            "mae": float(np.abs(Y[te] - p).mean()),
            "rmse": float(np.sqrt(((Y[te] - p) ** 2).mean())),
        }

    # Applicability domain: the full model's own hat matrix, from TRAIN rows.
    m = out["full"]
    Z = (D[tr] - m["mu"]) / m["sd"]
    p = Z.shape[1]
    G = np.linalg.pinv(Z.T @ Z + m["lam"] * np.eye(p))
    out["hat_inv"] = G
    out["h_star"] = float(3.0 * p / len(tr))
    h_tr = np.einsum("ij,jk,ik->i", Z, G, Z)
    out["h_train_flag_rate"] = float((h_tr > out["h_star"]).mean())
    out["h_train_p99"] = float(np.percentile(h_tr, 99))
    return out


def save(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    flat = {"hat_inv": model["hat_inv"]}
    meta = {k: model[k] for k in
            ("names", "n_total", "n_train", "n_holdout", "seed",
             "y_mean", "y_sd", "y_min", "y_max",
             "h_star", "h_train_flag_rate", "h_train_p99")}
    for tag in ("full", "comp"):
        m = model[tag]
        for k in ("mu", "sd", "coef"):
            flat[f"{tag}_{k}"] = m[k]
        meta[tag] = {k: m[k] for k in
                     ("cols", "intercept", "lam", "r2", "mae", "rmse")}
    flat["_meta_json"] = json.dumps(meta)
    np.savez(path, **flat)
    return path


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["_meta_json"]))
    model = {k: meta[k] for k in meta if k not in ("full", "comp")}
    model["hat_inv"] = z["hat_inv"]
    for tag in ("full", "comp"):
        model[tag] = dict(meta[tag])
        for k in ("mu", "sd", "coef"):
            model[tag][k] = z[f"{tag}_{k}"]
    return model


# ---------------------------------------------------------------- prediction

def leverage(model, D):
    """(M,) leverage h_i for a descriptor block, in the full model's geometry."""
    m = model["full"]
    Z = (np.asarray(D, float) - m["mu"]) / m["sd"]
    return np.einsum("ij,jk,ik->i", Z, model["hat_inv"], Z)


def predict_rows(model, D):
    """(alpha_hat, alpha_comp, leverage) for a (M, 20) descriptor block."""
    D = np.asarray(D, float)
    res = {}
    for tag in ("full", "comp"):
        m = model[tag]
        cols = list(m["cols"])
        res[tag] = _apply(m["mu"], m["sd"], m["coef"], m["intercept"], D[:, cols])
    return res["full"], res["comp"], leverage(model, D)


def predict_molecules(model, mols, atom_decoder=ATOM_DECODER):
    """Per-molecule readouts for MiDi `Molecule` objects.

    Returns a dict of (M,) arrays, NaN where the graph does not sanitise:
        alpha        alpha_hat, Bohr^3
        alpha_comp   composition-only prediction, Bohr^3
        n_heavy      heavy-atom count
        h            leverage; flagged above model["h_star"]
        valid        True where the graph sanitised
    """
    n = len(mols)
    out = {k: np.full(n, np.nan)
           for k in ("alpha", "alpha_comp", "n_heavy", "h")}
    out["valid"] = np.zeros(n, bool)
    rows, where = [], []
    for i, m in enumerate(mols):
        mol = rdmol_from_molecule(m, atom_decoder)
        if mol is None:
            continue
        rows.append(descriptors(mol))
        where.append(i)
        out["n_heavy"][i] = mol.GetNumHeavyAtoms()
        out["valid"][i] = True
    if rows:
        a, ac, h = predict_rows(model, np.vstack(rows))
        for k, v in (("alpha", a), ("alpha_comp", ac), ("h", h)):
            out[k][where] = v
    return out


# --------------------------------------------------------------- arm compare

def _paired(a, b):
    """Paired mean difference a-b over entries finite in BOTH, with se and t."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 2:
        return np.nan, np.nan, np.nan, n
    d = a[ok] - b[ok]
    se = float(d.std(ddof=1) / np.sqrt(n))
    return float(d.mean()), se, (float(d.mean() / se) if se > 0 else np.nan), n


def arm_table(readouts, control="control", h_star=None):
    """Rows of the per-arm polarisability table.

    readouts  {arm_name: dict from predict_molecules}, all arms sharing the
              same molecule ids in the same order, which is what makes the
              differences paired.

    Every difference is PAIRED against the control on the molecules valid in
    both arms, so `n_pair` can be smaller than either arm's valid count and
    differs from row to row. That is stated rather than hidden: the arms are not
    all being compared on the same subset.
    """
    ctrl = readouts[control]
    rows = []
    for name, r in readouts.items():
        per = r["alpha"] / r["n_heavy"]
        per_c = ctrl["alpha"] / ctrl["n_heavy"]
        d, se, t, npair = _paired(r["alpha"], ctrl["alpha"])
        dc, _, _, _ = _paired(r["alpha_comp"], ctrl["alpha_comp"])
        dp, sep, tp, _ = _paired(per, per_c)
        rows.append({
            "arm": name,
            "valid": int(np.nansum(r["valid"])),
            "n": len(r["valid"]),
            "alpha": float(np.nanmean(r["alpha"])),
            "d_alpha": d, "se": se, "t": t,
            "alpha_per_heavy": float(np.nanmean(per)),
            "d_per_heavy": dp, "t_per_heavy": tp,
            "d_comp": dc,
            "d_beyond": (np.nan if not np.isfinite(d) or not np.isfinite(dc)
                         else d - dc),
            "n_pair": npair,
            "h_max": float(np.nanmax(r["h"])) if np.isfinite(r["h"]).any()
                     else np.nan,
            "n_hi_lev": (int(np.nansum(r["h"] > h_star))
                         if h_star is not None else -1),
        })
    return rows


def print_table(rows, model=None, control="control"):
    hdr = (f"{'arm':<10}{'valid':>7}{'alpha':>9}{'d_alpha':>9}{'se':>7}{'t':>7}"
           f"{'a/heavy':>9}{'d/heavy':>9}{'d_comp':>8}{'d_beyond':>10}"
           f"{'n_pair':>8}{'hi_lev':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f = lambda v, w, p=3: (f"{v:>{w}.{p}f}" if np.isfinite(v) else f"{'-':>{w}}")
        print(f"{r['arm']:<10}{str(r['valid'])+'/'+str(r['n']):>7}"
              f"{f(r['alpha'],9,2)}{f(r['d_alpha'],9)}{f(r['se'],7)}"
              f"{f(r['t'],7,2)}{f(r['alpha_per_heavy'],9)}"
              f"{f(r['d_per_heavy'],9,4)}{f(r['d_comp'],8)}"
              f"{f(r['d_beyond'],10)}{r['n_pair']:>8}"
              f"{(str(r['n_hi_lev']) if r['n_hi_lev'] >= 0 else '-'):>8}")
    print()
    print("Units are Bohr^3, the QM9 unit for alpha. All differences are PAIRED")
    print("against the control on molecules valid in BOTH arms, so n_pair varies")
    print("by row and the rows are not all on the same subset.")
    print()
    print("  d_alpha    polarisability shift vs control")
    print("  t          d_alpha / se. |t| > ~2 is a shift the pairing resolves;")
    print("             with ~30 molecules and 5 arms this is descriptive, not")
    print("             a corrected significance test.")
    print("  d_comp     the part of d_alpha predicted by the arm's change in")
    print("             atomic composition alone")
    print("  d_beyond   d_alpha - d_comp: the shift composition does NOT")
    print("             explain. This is the column that separates steering the")
    print("             property from just growing the molecule.")
    print("  hi_lev     molecules above the warning leverage h*. The surrogate")
    print("             extrapolates there and its quoted MAE does not hold.")
    if model is not None:
        m, c = model["full"], model["comp"]
        print()
        print(f"Leverage: h* = {model['h_star']:.4f}; on the training rows "
              f"{100 * model['h_train_flag_rate']:.1f}% exceed it.")
        print()
        print(f"Surrogate: held-out R2 {m['r2']:.4f}, MAE {m['mae']:.3f}, "
              f"RMSE {m['rmse']:.3f} Bohr^3 on {model['n_holdout']} QM9 molecules")
        print(f"           (composition-only model R2 {c['r2']:.4f}, "
              f"MAE {c['mae']:.3f}); sd(alpha) = {model['y_sd']:.2f} Bohr^3.")
        print("           MAE is the error on ONE molecule. The paired se above")
        print("           is the right uncertainty for d_alpha: the surrogate's")
        print("           bias is common to both halves of a pair and cancels.")


# ------------------------------------------------------- QM9 training rows

def build_training_rows(dataset, labels_npz, atom_decoder=ATOM_DECODER,
                        limit=None, verbose=True):
    """(D, Y, n_failed) from a processed QM9 split and its DFT labels.

    dataset      a torch_geometric dataset, indexed by dataset position, as
                 `datamodule.test_dataloader().dataset` gives
    labels_npz   path to the capture store's labels.npz, which carries
                 `dataset_pos` and `polarisability` in aligned order

    The join is on `dataset_pos`, which is the alignment qm9_labels.py exists to
    get right: CSV row order is shuffled and dataset order is not, so joining on
    row order silently mislabels every molecule.
    """
    z = np.load(labels_npz)
    pos, y = z["dataset_pos"], z["polarisability"].astype(float)
    if limit is not None:
        pos, y = pos[:limit], y[:limit]
    D, Y, bad = [], [], 0
    for p, yy in zip(pos, y):
        d = dataset[int(p)]
        n = int(np.asarray(d.x).reshape(-1).size)
        bond = np.zeros((n, n), dtype=np.int64)
        ei = np.asarray(d.edge_index)
        ea = np.asarray(d.edge_attr).reshape(-1)
        bond[ei[0], ei[1]] = ea
        mol = rdmol_from_graph(np.asarray(d.x).reshape(-1),
                               np.asarray(d.charges).reshape(-1),
                               bond, atom_decoder)
        if mol is None:
            bad += 1
            continue
        D.append(descriptors(mol))
        Y.append(yy)
    if verbose:
        print(f"built {len(Y)} training molecules, {bad} failed to sanitise")
    return np.vstack(D), np.asarray(Y, float), bad


def fit_or_load(path, dataset=None, labels_npz=None,
                atom_decoder=ATOM_DECODER, refit=False, limit=None):
    """Load the cached surrogate, or build it once and cache it."""
    if os.path.exists(path) and not refit:
        return load(path), False
    if dataset is None or labels_npz is None:
        raise ValueError(f"no cached surrogate at {path}; pass dataset and "
                         "labels_npz to fit one")
    D, Y, _ = build_training_rows(dataset, labels_npz, atom_decoder, limit=limit)
    model = fit(D, Y)
    save(model, path)
    return model, True
