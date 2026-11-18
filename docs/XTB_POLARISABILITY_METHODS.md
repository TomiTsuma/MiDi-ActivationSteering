# xTB polarisability: method, validation, and the geometry decision

Written 2026-11-17. Everything numeric here was measured on this project's own
files, not taken from literature. Reproduction commands are at the end.

## 1. Why this exists

Every property claim in this study so far rests on **Crippen molar refractivity
(MolMR)**, an RDKit *graph descriptor* computed from the generated graph. It is
a proxy for polarisability via the Lorentz-Lorenz relation, it is extensive, and
it correlates with molecule size — the exact confound that invalidated the first
analysis. The first question a reader or examiner asks is whether a *physical*
estimate agrees.

GFN2/PTB-xTB provides one, post hoc, from the saved `chunk_*.pt` files. Nothing
is regenerated: each chunk already stores `atom_types`, `charges`, `bond_types`
and `positions`.

## 2. Which xTB number — the obvious choice is wrong

xtb reports two quantities called polarisability. They are **not**
interchangeable.

| quantity | how obtained | what it is |
|---|---|---|
| `Mol. α(0) /au` | `--gfn 2 --verbose`, dispersion section | the D4 model's sum of **charge-scaled atomic** polarisabilities |
| **isotropic dipole polarisability** | **`--ptb --alpha`** | a **numerical response** polarisability tensor |

`Mol. α(0)` is additive by construction. Verified on water:

```
#   Z          covCN         q      C6AA      α(0)
1   8 O        1.611    -0.564    24.329     6.657
2   1 H        0.806     0.282     0.778     1.386
3   1 H        0.806     0.282     0.778     1.386
Mol. α(0) /au  :  9.428780          # = 6.657 + 1.386 + 1.386, exactly
```

So `Mol. α(0)` is a composition-and-charge sum. **Using it as the independent
physical check would rebuild the very size confound the check exists to test.**
It is not used here.

`--ptb --alpha` instead returns a numerical polarisability tensor and its
isotropic trace, and it is the only readout in this study that reads the 3-D
coordinates at all. Water:

```
Numerical polarizability tensor: (a.u.)
     X    5.7436   -0.0000    0.0000
     Y   -0.0000    8.4129    0.0000
     Z    0.0000    0.0000    7.4812
Total isotropic dipole polarizability (a.u. / Å³):  7.2126   1.0688
```

**Units.** a.u. of polarisability is Bohr³, which is the unit QM9 reports `alpha`
in. No conversion is applied anywhere.

## 3. Validation against the DFT ground truth

395 labelled QM9 test molecules, scored at their **DFT geometries**, compared to
the QM9 `alpha` label. R²(affine) allows a best linear rescale, so a pure
unit/scale difference is not counted against a readout. All 395 converged.

| readout | Pearson | Spearman | R²(affine) | MAE (Bohr³) |
|---|---|---|---|---|
| **xTB PTB α** | **+0.9824** | **+0.9783** | **0.9651** | **1.25** |
| descriptor surrogate (`midi/alpha_surrogate.py`) | +0.9740 | +0.9731 | 0.9486 | 1.29 |
| Crippen MolMR | +0.9519 | +0.9489 | 0.9061 | 1.91 |
| heavy-atom count | +0.5585 | +0.4634 | 0.3119 | 5.70 |

For scale, sd(α) = 8.13 and mean(α) = 75.25 Bohr³. xTB mean 78.13, a ratio of
**1.038**, so it reads directly as Bohr³ without rescaling.

**Conclusion:** PTB α is the most accurate of the four *and* the only physically
independent one. The MolMR-based conclusions are not being checked against a
weaker instrument.

## 4. The geometry decision — this is the part that changed

The plan recorded "score as-is, because as-is is the honest match to what the
model produced." That reasoning is right about what as-is *means*, but as-is
**alone** is not defensible, for a measured reason.

PTB α is acutely and **asymmetrically** sensitive to coordinate error. Measured
by jittering the DFT geometries of 120 labelled molecules with isotropic
Gaussian noise and rescoring (paired, same molecules):

| geometry | Pearson vs DFT α | paired shift vs as-is (Bohr³) |
|---|---|---|
| as-is (DFT geometry) | +0.9791 | — |
| xtb-relaxed | +0.9831 | −1.16 ± 0.52 |
| jitter 0.05 Å | +0.9534 | +1.15 ± 1.98 |
| **jitter 0.15 Å** | **+0.7084** | **+15.95 ± 9.64 (max 53.4)** |
| jitter 0.15 Å → relax | +0.9830 | −1.15 ± 0.53 |

Three facts follow:

1. **Coordinate error inflates α upward.** 0.15 Å of noise adds ~16 Bohr³, a
   21 % rise on a mean of 75.
2. **It destroys the correlation with truth**, 0.98 → 0.71.
3. **Relaxation repairs it completely.** Jitter-then-relax lands within
   0.01 Bohr³ of relaxing the clean geometry, so relaxation recovers the same
   structure regardless of the noise it started from.

**Why this matters here and is not academic.** Steering degrades geometry, and
degrades it most at large |c|. Scoring as-is alone would therefore produce a
*positive* α shift concentrated in the most heavily steered arms — the **same
sign as the effect being claimed** — and that shift would be a geometry-quality
artefact, not a property change. A reviewer who knows this literature will ask.

**Decision: score both, report both, per molecule.**

| column | meaning |
|---|---|
| `alpha_asis` | the model's own output, geometry included, warts and all |
| `alpha_opt` | PTB α after GFN2 `--opt tight`: the nearest physically reasonable molecule |
| `rmsd_opt` | Kabsch-aligned RMSD the relaxation moved the atoms (Å) |

`rmsd_opt` is the diagnostic that ties them together: a direct per-molecule
measure of how unphysical the generated coordinates were. **If
`alpha_asis − alpha_opt` tracks `rmsd_opt` within an arm, the as-is number is
reporting geometry quality, not polarisability.** Kabsch alignment is used
because xtb may translate or reorient a structure during optimisation, and a raw
coordinate difference would report that motion as deformation.

The honest headline claim is whichever of the two is *less* favourable to the
steering hypothesis, with the other reported beside it.

## 5. Charged molecules — a correlated-dropout trap, fixed

MiDi generates graphs with formal charges. A nonzero net charge often makes the
electron count odd, and PTB assumes a closed-shell singlet, so it aborts:

```
-1- ptb_calculator_singlepoint: Number of electrons and spin multiplicity do not match.
```

In the first 20-molecule test this silently killed **3 of 20**, all of them ions
(net −1, +1, +2). Dropping them would be a **correlated dropout**: if steered
arms produce more ions than the control — which is exactly what degradation
looks like — the arm comparison is biased by whichever way ions differ.

Fix: pass `--uhf` from the electron parity,

```
n_elec = Σ Z(atom) − net_charge        # Z: H 1, C 6, N 7, O 8, F 9
uhf    = n_elec mod 2
```

After the fix, **20/20 converge**, including all six charged molecules. This is
`uhf_for()` in `midi/xtb_runner.py`, and `uhf` is recorded per row so the
assumption is auditable rather than buried.

## 6. Other correctness details

- **Restart-file isolation.** xtb writes `xtbrestart` into its working
  directory and will *read* a stale one as an SCF guess, which would make one
  molecule's result depend on whichever ran before it. Every call runs in its
  own temporary directory and passes `--norestart`.
- **Threads.** `OMP_NUM_THREADS=1` per worker; parallelism is across molecules.
- **Cost.** ~8 ms per single point; ~190 ms per molecule for both modes
  including the relaxation. Measured throughput 4.5 molecules/s on 2 workers,
  so the 11,500-molecule sweep takes ~42 min. The plan's "about one second per
  molecule, roughly 75 minutes" was conservative by an order of magnitude on
  the single point.
- **Resumability.** One CSV per input chunk, written to a temp name and
  renamed, mirroring the input tree. An interrupted run loses at most one chunk
  and a rerun skips everything already finished — the same convention as the
  rest of the pipeline.

## 7. Reproduction

xtb 6.7.1, official Linux x86_64 release build from the grimme-lab GitHub
releases page.

```bash
# score the 23-arm sweep, both geometry modes
python scripts/score_xtb.py --runs results/sweep --out results/xtb \
    --mode both --workers 4

# aggregate
python scripts/score_xtb.py --out results/xtb --summary
```

Per-row output columns: `arm, chunk, mol_id, n_atoms, n_heavy, net_charge,
alpha_asis, alpha_asis_ang3, status_asis, alpha_opt, status_opt, rmsd_opt, uhf,
ms`.

## 8. What still has to be checked once scoring finishes

1. **Convergence rate per arm.** If it differs systematically across arms, the
   surviving molecules are a biased sample and that must be stated.
2. **`rmsd_opt` vs |c|.** Quantifies geometry degradation independently of the
   property, and is the test for whether `alpha_asis` is contaminated.
3. **xTB α vs MolMR on the control arm.** The plan's step 3. Near-zero
   correlation would mean one of the two is wrong; given §3, MolMR would be the
   suspect.
4. **Do the as-is and relaxed conclusions agree?** If they do, the finding is
   robust. If they do not, the geometry artefact is the finding.

---

# 9. RESULTS — sweep scored 2026-11-17

11,500 molecules, 23 arms, both geometry modes, 44.1 min wall on 2 workers.

**Convergence: 11,493 / 11,500 as-is and 11,488 / 11,500 relaxed.** Worst arm
497/500. The `--uhf` parity fix is what bought this; without it every ion failed,
and 17 % of control molecules are ions. There is no correlated dropout, so no arm
is being compared on a biased subset.

## 9.1 Headline: xTB confirms the null, more decisively than MolMR did

**Sign test.** A direction that encodes a property moves it up at +c and down at
−c. On relaxed α per heavy atom:

| \|c\| | +c (t) | −c (t) | verdict |
|---|---|---|---|
| 10 | +0.0113 (+0.67) | +0.0206 (+2.25) | same sign |
| 20 | +0.1695 (+3.54) | +0.0378 (+0.96) | same sign |
| 40 | −0.0254 (−0.60) | −0.0705 (−1.15) | same sign |
| 80 | −0.0111 (−0.22) | +0.0797 (+0.53) | "opposite", but both t < 1 — noise |
| 160 | +0.0034 (+0.08) | +0.0118 (+0.14) | same sign |

Four of five magnitudes same sign; the one reversal is statistically
indistinguishable from zero on both arms. **No directional control.**

**The controls match the treatment.** Raw paired Δα, relaxed:
`dom_late_cp40` +2.883 (t 6.84), `rand_late_cp40` +2.736 (t 6.41),
`shuf_late_cp80` +2.959 (t 8.24). Directions carrying **no property
information** produce the same "highly significant" α rise. The large t values
are perturbation damage, not steering.

**The means are outlier-driven.** `dom_late_cp40`: mean Δα +2.883, **median
+0.005**. Most molecules are unchanged and a minority blow up — the signature of
damage, not a distributional shift in a property.

**The one standout does not survive.** `dom` at c=+20 gives +0.1695 per heavy
atom (t 3.54) and is outside the rand/shuf range at that magnitude. But it
vanishes at c=40, 80 and 160, and `shuf_late_cp40` reaches +0.1763 (t 2.40) —
the same size from a shuffled direction. A real property direction does not
appear at one coefficient and disappear at larger ones.

## 9.2 The as-is geometry artefact, now measured on the real data

Scoring as-is only would have produced a **false positive with a dose-response
shape**. Δα per heavy atom for the dom +c arms:

| c | as-is | relaxed |
|---|---|---|
| +10 | −0.0658 | +0.0113 |
| +20 | **+0.1551** | +0.1695 |
| +40 | **+0.1554** | −0.0254 |
| +80 | **+0.1192** | −0.0111 |
| +160 | **+0.1031** | +0.0034 |

As-is shows a clean positive plateau of +0.10 to +0.16 across four consecutive
coefficients — exactly what a working steering method looks like. Relaxation
erases it. (`rand_late_cp40` as-is reaches +0.5226 per heavy, larger than every
dom arm, which independently exposes it as an artefact.)

This vindicates scoring both. **Reporting as-is alone would have put a
geometry-quality artefact in the paper as a property effect.**

Magnitude of the artefact over all 11,500: as-is minus relaxed mean +2.403,
median +1.777, p95 +15.38, max +1247.6 Bohr³; **8.6 % of molecules differ by
more than 10 Bohr³.**

## 9.3 Geometry degradation is real and scales with |c|

Relaxation RMSD (Å), which is a property-free damage metric:

| arm | mean | median | frac > 0.5 Å |
|---|---|---|---|
| control | 0.268 | 0.154 | 0.164 |
| dom ±10 | 0.275 / 0.277 | 0.161 / 0.173 | 0.164 / 0.162 |
| dom ±20 | 0.362 / 0.441 | 0.224 / 0.343 | 0.246 / 0.346 |
| dom ±40 | 0.538 / 0.356 | 0.427 / 0.242 | 0.402 / 0.248 |
| dom ±80 | 0.647 / 0.361 | 0.463 / 0.231 | 0.482 / 0.250 |
| dom ±160 | 0.608 / 0.327 | 0.490 / 0.206 | 0.490 / 0.218 |

Spearman(|c|, mean RMSD) = **+0.548** over the 22 steered arms. Degradation is
stronger on the −c side. This is an independent, property-free confirmation that
the injection damages the generation, and it is worth a figure.

## 9.4 Step 3 — xTB vs MolMR on 490 control molecules

| pair | Pearson | Spearman |
|---|---|---|
| xTB α_opt vs MolMR | +0.710 | +0.806 |
| xTB α_asis vs MolMR | +0.592 | +0.791 |
| xTB α_opt vs descriptor surrogate | +0.750 | +0.850 |
| xTB α_opt vs heavy count | +0.488 | +0.480 |

Not near zero, so the two readouts agree — but far more weakly than MolMR's
+0.952 against DFT α on *real* QM9 molecules (§3). Since xTB is the more
accurate of the two there, **MolMR is the degraded party on generated
molecules**. 17 % of them carry a formal charge, and Crippen's parameters are
fitted on neutral drug-like molecules. This is a caveat for the wider practice of
evaluating generative molecular models with RDKit descriptors, not only for this
pipeline.

## 9.5 What to claim

The null is robust under a physically independent, DFT-calibrated estimator
(R² 0.965, Spearman 0.978 against the QM9 labels), at n=500 per arm rather than
32, with negative controls that reproduce the treatment's apparent effect. The
honest statement is that **steering along `d` or `w` at layer 10 does not produce
directional control of polarisability, and the apparent raw-α rises are
explained by geometry degradation plus outliers.**

Data: `results/xtb/` (230 per-chunk CSVs), `xtb_sweep_rows.csv` (all 11,500
rows), `xtb_arm_table.csv` (arm-level paired statistics).
