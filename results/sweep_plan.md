# The main dose-response sweep, with the Section 3.8 controls

Decided 2026-11-14. Verified by dry run at commit 40d020b. Site: layer 10,
channel X, block `X_mean`, window t in [50, 100], property polarisability,
direction = composition-residual difference of means.

## The design

23 cells, one invocation, 500 molecules each, 11,500 total, about 14.8 h.

| arm | coefficients | cells | what it is |
|---|---|---|---|
| `dom` | +-10, +-20, +-40, +-80, +-160 | 10 | the real direction; this is the grid from `coefficient_grid.md` |
| `rand` | +-20, +-40, +-80 | 6 | seeded unit-sphere draw at the real arm's norm |
| `shuf` | +-20, +-40, +-80 | 6 | difference of means from permuted labels, also at the real arm's norm |
| `control` | 0 | 1 | shared, generated once |

The real arm carries the full curve because the response is non-monotone and
the turnover sits between +80 and +160. The controls carry only the points
where a claim will be made: +-20 is the onset of degradation, +-40 the widest
signed asymmetry, +-80 the peak raw effect.

**All in one invocation, which is not a convenience.** Every cell uses ids
0..499 and run_seed 1234, so molecule i is the same draw everywhere and the
only difference is the injection. Run as a second sweep, the controls would
differ from the real arm in the ids, the seed, the wall clock and possibly
the direction file, and a difference in outcome could be any of them. As a
side effect, ids 0..199 are the same molecules as the existing pilot, so that
pilot is a subset-compatible preview rather than a separate experiment.

## Matched norm

The injected vector is `coeff * N[t] * V[t]` with V a unit vector, so its
length is `|coeff * N[t]|` whatever V happens to be. The controls replace V
and keep N = nrm_dom, so the injected norm is identical to the real arm's at
every timestep by construction: `coeff * 0.9964` at t=50 in all 23 cells.
Checked before any compute, and again in `test_control_injects_at_matched_norm`.

## The finding that changes how 3.8 must be written

**The two controls are not equivalent nulls.** A permuted split is still a
difference between the means of two halves of the same activations, so
d_shuf is a random vector drawn with the DATA's covariance, not an isotropic
one. For any fixed unit u,

    rms cos(d_shuf, u) = sqrt( u' Sigma u / trace(Sigma) )

Measured at layer 10, t=50 over 300 permutations:

| draw | predicted rms cos | observed rms | observed mean abs | P(abs > 0.25) |
|---|---|---|---|---|
| unit sphere | 0.0625 | 0.0622 | 0.0504 | 0.000 |
| permuted labels | 0.4069 | 0.3969 | 0.3353 | 0.593 |

The real direction lies along 16.6% of the activation variance, where
isotropy would give 0.39%, and needs only 2 principal components for 90% of
its length. That is why the shuffled direction is six times more aligned
with it than chance.

Consequences:

- `rand` is the **clean null**: isotropic, 0.06, and 300 draws never exceeded
  0.21. A trip above 0.25 means a bug, so it carries a hard gate.
- `shuf` is a **conservative control**: it already carries about 0.40 of the
  real direction's orientation while holding only 3.4% of its length. The
  real arm has to beat a direction that is partly itself. If the real arm
  wins, the content claim is strong. If they tie, that is ambiguous between
  "no content" and "the shared 40% does the work", and the random arm is
  what separates those.
- `shuf` gets **no rejection gate**. Discarding a draw for coming out high
  would be selecting controls by their outcome.
- The formula is a ratio of expectations and holds only while the variance
  share is small: 2.5% error at a share of 0.17, 19% at 0.67. Do not quote it
  at a site where the share is above about 0.4.

## Analysing the signs

The sign is meaningful for `dom` only. A random unit vector and its negation
are both random draws, and a label permutation has no privileged
orientation, so +c and -c on a control arm are **two independent draws at the
same injected norm**, not a signed response curve. Pool the control arms by
|c|, which also gives them n = 1000 per magnitude against the real arm's 500.

## What to read first as arms land

`mean_heavy` in `summary.csv`. The scan showed the real arm raising mean
heavy-atom count from 8.82 to 9.28 at c = +80, and `coefficient_grid.md`
established that the apparent property shift is that size change. If the
random arm at matched norm moves size the same way, the size effect is a
generic consequence of inflating activations and the apparent property shift
is explained outright. If it leaves size alone, the real arm's size effect is
content-driven. That is a result from `mean_heavy` alone, before any xTB.

## xTB is off the critical path

Each chunk stores `atom_types`, `charges`, `bond_types` and `positions`, so
GFN2-xTB can score the saved `.pt` files afterwards without regenerating a
molecule. It is not a launch blocker. The primary endpoint must still be
size-adjusted: report dMolMR, dMolMR per heavy atom and dHeavy together.

## Budget

778 mol/h, the measured STEERED rate from `results/pilot` (2,000 molecules in
9,250 s of summed arm wall clock). The 1,046 mol/h in Section 3.11 is the
unsteered generator and overstates the budget by a third.

11,500 molecules / 778 = 14.8 h. The item's "a few days" was off by a factor
of five, which is why n = 500 is affordable and its fallback to 300 is not
needed.

## Command

    cd C:\Users\tsuma.thomas\Documents\MiDi-ActivationSteering\midi
    $R="C:\Users\tsuma.thomas\Documents\MiDi-ActivationSteering"
    python3.11 -u ..\scripts\run_pilot.py `
      dataset=qm9 dataset.remove_h=False +experiment=qm9_with_h_adaptive `
      general.wandb=disabled general.gpus=1 `
      "general.test_only=$R\midi\checkpoints\checkpoint_qm9_h_adaptive.ckpt" `
      "+pilot.arms=[dom,rand,shuf]" `
      "+pilot.coeffs=[-160,-80,-40,-20,-10,10,20,40,80,160]" `
      "+pilot.control_coeffs=[-80,-40,-20,20,40,80]" `
      +pilot.n=500 +pilot.window=late `
      "+pilot.store=$R\activations_test_full" `
      "+pilot.dirs=$R\results\directions" `
      "+pilot.out=$R\results\sweep"

Add `+pilot.dry_run=true` to print the plan and the control diagnostics
without generating. Resumable: completed arms are skipped and chunks are
written atomically, so a crash costs at most 50 molecules.
