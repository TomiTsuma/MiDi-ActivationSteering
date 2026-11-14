"""scripts/run_pilot.py -- the steering pilot: matched-seed arms, resumable.

    python3.11 -u scripts/run_pilot.py <store> <dirs> <out> [--prop P] [--n 200]

Arms (the default set, 10 of them, about 2 hours at the measured 1,046 mol/h):

    control                       coeff 0, no hooks attached at all
    dom  late  x 3 coefficients   difference of means, composition-residual
    dom  early x 3 coefficients   the same direction, earlier window
    probe late x 3 coefficients   the ridge probe's own direction

Why the probe arm exists: measured on this store, cos(probe w, diff-of-means)
is only 0.03 to 0.20. The two candidate directions are nearly orthogonal, so
which one to steer with is an open question, not a detail. Three arms answer
it for the price of 35 minutes.

Coefficients are not fixed numbers. ||d|| varies threefold between properties
and the activation norm is about 7.76, so a fixed grid is neither comparable
across properties nor guaranteed to perturb anything. Each arm is placed at a
target ALIGNMENT (0.2, 0.5, 0.8) and the coefficient is solved for, so the
arms sit at matched angle and the norm ratios can be compared.

Every arm uses the same molecule ids and the same run_seed, so molecule i is
the same draw in every arm and the only difference is the injection. That
pairing is what Block 1's keyed RNG bought.

Resumable: chunks are written atomically per arm and a restart skips arms
already complete.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import hydra
import omegaconf

import midi._compat  # noqa: F401
from midi import direction, direction_calib, rng, steer
from midi.generate import _setup, _atomic_save, _count_valid, N_MAX

LAYER = 10
BLOCK = "X_mean"
CHANNEL = "X"
WINDOWS = {"late": (50, 100), "early": (200, 300)}
TARGET_COS = (0.2, 0.5, 0.8)
CHUNK = 50

SUMMARY_FIELDS = [
    "arm", "direction", "window", "t_lo", "t_hi", "target_cos", "coeff",
    "n", "valid", "valid_pct", "mean_heavy", "mean_atoms",
    "rho_mean", "cos_mean", "n_injected", "seconds",
]


def heavy_and_atoms(mols, atom_decoder):
    h_i = list(atom_decoder).index("H") if "H" in atom_decoder else -1
    n_atoms, n_heavy = [], []
    for m in mols:
        a = m.atom_types.cpu().numpy()
        n_atoms.append(len(a))
        n_heavy.append(int((a != h_i).sum()) if h_i >= 0 else len(a))
    return float(np.mean(n_heavy)), float(np.mean(n_atoms))


def run_arm(model, device, arm, out_dir, ids, seed, state, atom_decoder):
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = (len(ids) + CHUNK - 1) // CHUNK
    done = {i for i in range(n_chunks)
            if os.path.exists(os.path.join(out_dir, f"chunk_{i:05d}.pt"))}
    t0 = time.time()
    handles = steer.attach(model, state) if state is not None else []
    mols_all = []
    try:
        for ci in range(n_chunks):
            path = os.path.join(out_dir, f"chunk_{ci:05d}.pt")
            lo, hi = ci * CHUNK, min((ci + 1) * CHUNK, len(ids))
            batch = ids[lo:hi]
            if ci in done:
                continue
            if state is not None:
                state.reset()
            n_nodes = rng.sample_n_nodes(seed, batch, model.node_dist.prob)
            with torch.no_grad():
                mols = model.sample_batch(
                    n_nodes=n_nodes.tolist(), mol_ids=batch, run_seed=seed,
                    n_max_fixed=N_MAX, test=True, save_final=0, keep_chain=0,
                    visualize=False, batch_id=ci)
            _atomic_save({
                "mol_ids": list(batch),
                "atom_types": [m.atom_types.cpu() for m in mols],
                "charges": [m.charges.cpu() for m in mols],
                "bond_types": [m.bond_types.cpu() for m in mols],
                "positions": [m.positions.cpu() for m in mols],
            }, path, torch_save=True)
            mols_all.extend(mols)
            print(f"    chunk {ci + 1}/{n_chunks}  {len(mols)} mols  "
                  f"{time.time() - t0:.0f}s", flush=True)
    finally:
        steer.detach(handles)

    if not mols_all:
        return None            # fully resumed; nothing recomputed

    rho = cos = float("nan")
    n_inj = 0
    if state is not None:
        pm = state.per_molecule()
        if pm is not None:
            rho, cos = float(np.mean(pm[0])), float(np.mean(pm[1]))
        n_inj = state.n_injected
        with open(os.path.join(out_dir, "diag_steps.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "rho_mean", "cos_mean",
                                              "norm_unsteered_mean"])
            w.writeheader()
            for r in state.steps:
                w.writerow(r)

    valid = _count_valid(mols_all)
    mh, ma = heavy_and_atoms(mols_all, atom_decoder)
    return {
        "n": len(mols_all), "valid": valid,
        "valid_pct": round(100.0 * valid / max(len(mols_all), 1), 2),
        "mean_heavy": round(mh, 3), "mean_atoms": round(ma, 3),
        "rho_mean": round(rho, 5), "cos_mean": round(cos, 5),
        "n_injected": n_inj, "seconds": round(time.time() - t0, 1),
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg):
    from hydra.utils import get_original_cwd
    orig = get_original_cwd()

    # Defaults, overridable with +pilot.<key>=<value>. Relative paths resolve
    # against the ORIGINAL cwd, not Hydra's run directory.
    with omegaconf.open_dict(cfg):
        cfg.pilot = omegaconf.OmegaConf.merge(
            omegaconf.OmegaConf.create({
                "store": "activations_test_full",
                "dirs": "results/directions",
                "out": "results/pilot",
                "prop": "polarisability",
                "n": 200,
                "run_seed": 1234,
                "dry_run": False,   # print the plan and the calibration, generate nothing
                # Explicit SIGNED coefficients. When set, this replaces the
                # auto-calibrated three-target plan with one arm per value,
                # which is what a degradation ladder needs: the calibration
                # targets alignment, and past cos ~0.9 the angle saturates
                # while only the norm keeps climbing, so alignment stops being
                # a useful axis exactly where degradation starts.
                "coeffs": None,
                "window": "late",
                "direction": "dom",
            }),
            cfg.get("pilot", {}) or {})

    def abspath(p):
        return p if os.path.isabs(p) else os.path.abspath(os.path.join(orig, p))

    store = abspath(cfg.pilot.store)
    dirs = abspath(cfg.pilot.dirs)
    out = abspath(cfg.pilot.out)
    prop = cfg.pilot.prop
    n_mol = int(cfg.pilot.n)
    seed = int(cfg.pilot.run_seed)
    os.makedirs(out, exist_ok=True)

    with omegaconf.open_dict(cfg):
        cfg.gen = omegaconf.OmegaConf.create(
            {"mode": "generate", "target": n_mol, "chunk": CHUNK, "out": out,
             "run_seed": seed, "metrics_at_end": False, "metrics_device": "cpu"})
    device, dinfos, model, _ = _setup(cfg)
    atom_decoder = dinfos.atom_decoder
    ids = list(range(n_mol))

    res = direction.load(os.path.join(dirs, f"dir_L{LAYER}_{BLOCK}_{prop}.npz"))
    ts = np.asarray(res["timesteps"])
    A = direction.load_block(store, LAYER, int(np.where(ts == 50)[0][0]), BLOCK)

    # Per-timestep directions, both flavours.
    D_dom = np.asarray(res["d_resid"])
    nrm_dom = np.linalg.norm(D_dom, axis=1)
    V_dom = D_dom / nrm_dom[:, None]
    w = np.asarray(res["w_probe_raw"])
    V_probe = np.repeat((w / np.linalg.norm(w))[None, :], len(ts), axis=0)
    nrm_probe = np.full(len(ts), float(np.linalg.norm(D_dom[list(ts).index(50)])))

    plan = [{"arm": "control", "direction": "none", "window": "-",
             "target_cos": 0.0, "coeff": 0.0}]

    explicit = cfg.pilot.coeffs
    if explicit is not None:
        dname = str(cfg.pilot.direction)
        wname = str(cfg.pilot.window)
        V = V_dom if dname == "dom" else V_probe
        N = nrm_dom if dname == "dom" else nrm_probe
        vref, nref = V[list(ts).index(50)], float(N[list(ts).index(50)])
        for c in [float(x) for x in explicit]:
            rho, cos = direction_calib.rho_and_cos(A, vref, c * nref)
            plan.append({"arm": f"{dname}_{wname}_c{c:+g}".replace("+", "p")
                                                          .replace("-", "m"),
                         "direction": dname, "window": wname,
                         "target_cos": float(np.median(cos)),
                         "coeff": c,
                         "_rho_pred": round(float(np.median(rho)), 4)})

    auto = [] if explicit is not None else [("dom", V_dom, nrm_dom),
                                            ("probe", V_probe, nrm_probe)]
    for dname, V, N in auto:
        wins = WINDOWS.keys() if dname == "dom" else ["late"]
        vref = V[list(ts).index(50)]
        picks = direction_calib.pick_coeffs(A, vref, float(N[list(ts).index(50)]),
                                            TARGET_COS)
        for wname in wins:
            for p in picks:
                plan.append({"arm": f"{dname}_{wname}_cos{p['target_cos']}",
                             "direction": dname, "window": wname,
                             "target_cos": p["target_cos"],
                             "coeff": round(p["coeff"], 4),
                             "_rho_pred": round(p["rho_median"], 4)})

    print(f"[pilot] property   {prop}")
    print(f"[pilot] site       layer {LAYER}, channel {CHANNEL}, block {BLOCK}")
    print(f"[pilot] molecules  {n_mol} per arm, ids 0..{n_mol - 1}, seed {seed}")
    print(f"[pilot] arms       {len(plan)}  -> {len(plan) * n_mol:,} molecules, "
          f"about {len(plan) * n_mol / 1046.0:.1f} h at the measured rate")
    print(f"[pilot] cos(probe w, dom) = {res['cos_w_vs_d_resid']:+.3f}: the two "
          f"directions are nearly orthogonal, which is why both are run")
    for p in plan:
        print(f"    {p['arm']:<28} coeff {p['coeff']:>8}  "
              f"predicted rho {p.get('_rho_pred', 1.0)}")
    print()

    if bool(cfg.pilot.dry_run):
        print("[pilot] dry run: plan and calibration above, nothing generated")
        return

    sum_path = os.path.join(out, "summary.csv")
    seen = set()
    if os.path.exists(sum_path):
        with open(sum_path, newline="") as f:
            seen = {r["arm"] for r in csv.DictReader(f)}
        print(f"[pilot] RESUMING: {len(seen)} arms already summarised")

    fresh = not os.path.exists(sum_path)
    with open(sum_path, "a", newline="") as fh:
        w_csv = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if fresh:
            w_csv.writeheader()
            fh.flush()
        for p in plan:
            if p["arm"] in seen:
                print(f"[pilot] {p['arm']}: done")
                continue
            print(f"[pilot] {p['arm']}  coeff={p['coeff']}", flush=True)
            state = None
            t_lo = t_hi = 0
            if p["direction"] != "none":
                t_lo, t_hi = WINDOWS[p["window"]]
                V = V_dom if p["direction"] == "dom" else V_probe
                N = nrm_dom if p["direction"] == "dom" else nrm_probe
                state = steer.SteerState(
                    layer=LAYER, channel=CHANNEL, v=V, d_norm=N,
                    coeff=p["coeff"], window=(t_lo, t_hi), T=model.T,
                    device=device, v_timesteps=ts)
            r = run_arm(model, device, p["arm"], os.path.join(out, p["arm"]),
                        ids, seed, state, atom_decoder)
            if r is None:
                print(f"    (all chunks already on disk; no summary written)")
                continue
            row = {"arm": p["arm"], "direction": p["direction"],
                   "window": p["window"], "t_lo": t_lo, "t_hi": t_hi,
                   "target_cos": p["target_cos"], "coeff": p["coeff"]}
            row.update(r)
            w_csv.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
            print(f"    valid {r['valid_pct']}%  heavy {r['mean_heavy']}  "
                  f"rho {r['rho_mean']}  cos {r['cos_mean']}  "
                  f"{r['seconds']}s", flush=True)

    with open(os.path.join(out, "pilot_manifest.json"), "w") as f:
        json.dump({"property": prop, "layer": LAYER, "block": BLOCK,
                   "channel": CHANNEL, "windows": WINDOWS,
                   "target_cos": list(TARGET_COS), "n_per_arm": n_mol,
                   "run_seed": seed, "store": store, "dirs": dirs,
                   "cos_probe_vs_dom": res["cos_w_vs_d_resid"],
                   "arms": [{k: v for k, v in p.items() if not k.startswith("_")}
                            for p in plan]}, f, indent=2)
    print(f"\n[pilot] done -> {sum_path}")


if __name__ == "__main__":
    main()
