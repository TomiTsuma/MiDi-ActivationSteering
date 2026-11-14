"""scripts/analyse_pilot.py -- read the pilot as PAIRED data.

    python3.11 -u scripts/analyse_pilot.py <pilot_dir>

Arm means waste the design. Every arm used the same molecule ids and the same
run_seed, so molecule i in a steered arm is the matched twin of molecule i in
the control and the only difference is the injection. The paired difference
has far less variance than the difference of means, and it also answers the
question the arm means cannot: did this molecule change AT ALL?

Property proxy: Crippen molar refractivity (MolMR). Polarisability and molar
refractivity are connected by the Lorentz-Lorenz relation, so MolMR is a
physically motivated stand-in rather than an arbitrary descriptor. It is a
PROXY, computed by RDKit from the graph, not a measurement; xTB is the right
tool for the sweep and this is the right tool for today.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")
ATOM_DECODER = ("H", "C", "N", "O", "F")


def load_arm(d):
    """{mol_id: dict} from an arm's chunk_*.pt files."""
    import torch
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("chunk_") and f.endswith(".pt")):
            continue
        z = torch.load(os.path.join(d, f), map_location="cpu", weights_only=False)
        for i, mid in enumerate(z["mol_ids"]):
            out[int(mid)] = {
                "atom_types": z["atom_types"][i].numpy(),
                "charges": z["charges"][i].numpy(),
                "bond_types": z["bond_types"][i].numpy(),
                "positions": z["positions"][i].numpy(),
            }
    return out


def to_mol(rec):
    """Build an RDKit mol from the stored graph. None if it will not sanitise."""
    bond_dict = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                 Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    m = Chem.RWMol()
    for a, c in zip(rec["atom_types"], rec["charges"]):
        at = Chem.Atom(ATOM_DECODER[int(a)])
        if int(c) != 0:
            at.SetFormalCharge(int(c))
        m.AddAtom(at)
    E = np.triu(rec["bond_types"], 1)
    for i, j in zip(*np.nonzero(E)):
        b = int(E[i, j])
        if 1 <= b <= 4:
            m.AddBond(int(i), int(j), bond_dict[b])
    try:
        mol = m.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def descriptors(rec):
    mol = to_mol(rec)
    if mol is None:
        return {"valid": 0, "molmr": np.nan, "molwt": np.nan,
                "heavy": np.nan, "rings": np.nan}
    return {"valid": 1,
            "molmr": float(Crippen.MolMR(mol)),
            "molwt": float(Descriptors.MolWt(mol)),
            "heavy": float(mol.GetNumHeavyAtoms()),
            "rings": float(Descriptors.RingCount(mol))}


def identical(a, b):
    return all(np.array_equal(a[k], b[k]) for k in
               ("atom_types", "charges", "bond_types", "positions"))


def main(pilot):
    pilot = os.path.abspath(pilot)
    with open(os.path.join(pilot, "summary.csv"), newline="") as f:
        arms = list(csv.DictReader(f))

    print(f"pilot {pilot}\n")
    ctrl = load_arm(os.path.join(pilot, "control"))
    ctrl_d = {mid: descriptors(r) for mid, r in ctrl.items()}
    print(f"control: {len(ctrl)} molecules, "
          f"{sum(v['valid'] for v in ctrl_d.values())} sanitise\n")

    hdr = (f"{'arm':<22}{'coeff':>8}{'rho':>7}{'cos':>7}"
           f"{'changed':>9}{'dMolMR':>9}{'se':>7}{'t':>7}"
           f"{'dHeavy':>8}{'valid':>7}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for a in arms:
        if a["arm"] == "control":
            continue
        arm = load_arm(os.path.join(pilot, a["arm"]))
        ids = sorted(set(arm) & set(ctrl))
        n_changed = sum(0 if identical(ctrl[i], arm[i]) else 1 for i in ids)

        dm, dh, nval = [], [], 0
        for i in ids:
            d = descriptors(arm[i])
            nval += d["valid"]
            c = ctrl_d[i]
            if d["valid"] and c["valid"]:
                dm.append(d["molmr"] - c["molmr"])
                dh.append(d["heavy"] - c["heavy"])
        dm, dh = np.asarray(dm), np.asarray(dh)
        se = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else np.nan
        tstat = dm.mean() / se if se and np.isfinite(se) and se > 0 else np.nan

        print(f"{a['arm']:<22}{float(a['coeff']):>8.2f}"
              f"{float(a['rho_mean']):>7.3f}{float(a['cos_mean']):>7.3f}"
              f"{100.0 * n_changed / len(ids):>8.0f}%"
              f"{dm.mean():>9.3f}{se:>7.3f}{tstat:>7.1f}"
              f"{dh.mean():>8.3f}{100.0 * nval / len(ids):>6.1f}%")
        rows.append({"arm": a["arm"], "coeff": float(a["coeff"]),
                     "rho": float(a["rho_mean"]), "cos": float(a["cos_mean"]),
                     "pct_changed": 100.0 * n_changed / len(ids),
                     "d_molmr": dm.mean(), "se": se, "t": tstat,
                     "d_heavy": dh.mean(),
                     "valid_pct": 100.0 * nval / len(ids)})

    print(f"\ncontrol MolMR: mean "
          f"{np.nanmean([v['molmr'] for v in ctrl_d.values()]):.3f}, sd "
          f"{np.nanstd([v['molmr'] for v in ctrl_d.values()]):.3f}")
    print("dMolMR is the PAIRED mean difference from control; t = dMolMR/se.")
    print("A |t| above about 2 is a detectable shift; the sign is the "
          "direction the property moved.")

    out = os.path.join(pilot, "paired_analysis.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/pilot")
