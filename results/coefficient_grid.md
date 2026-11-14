# Signed coefficient grid for the main sweep

Decided 2026-11-13 from `results/pilot` (10 arms, 200 molecules each) and
`results/scan` (9 arms, 50 each). Site: layer 10, channel X, block `X_mean`,
direction = composition-residual difference of means, window t in [50, 100].
Coefficients are in units of ||d||, the low-to-high gap, so c = 1 moves a
molecule by the full average gap between the property's halves.

## The grid

    c in { -160, -80, -40, -20, -10, 0, +10, +20, +40, +80, +160 }

Eleven arms. 400 molecules per arm, same ids and same run_seed in every arm,
so every molecule has a matched twin in every other arm.

## Why each point is there

| c | rho | cos | validity | what it anchors |
|---|---|---|---|---|
| 0 | 1.00 | - | 100% (n=50), 96.5% (n=200) | matched-seed baseline; the pairing is the whole design |
| ±10 | 1.68 | 0.80 | 97.5% | **the no-effect end.** Measured at c=10.77 on 200 molecules: validity unchanged, paired dMolMR t = 1.2. Nothing happens here and that is the point |
| ±20 | 2.75 / 2.81 | ±0.93 | 88% / 86% | **onset of degradation.** First validity drop, 100% -> 88%. Also the only size-independent property signal in the whole scan (dMR per heavy atom t = 3.2 at +20) |
| ±40 | 5.24 / 5.21 | ±0.98 | 88% / **52%** | **widest signed asymmetry.** The positive side is untouched while the negative side collapses to 52%. If the sweep drops one magnitude, it must not be this one |
| ±80 | 10.24 / 10.16 | ±0.995 | 84% / 62% | **peak raw effect** (dMolMR +0.735) and rho ~10 |
| ±160 | 20.36 / 20.13 | ±0.999 | 74% / 62% | **the clearly-degrading end.** The raw effect has turned over (+0.735 -> +0.385) while validity keeps falling. The response is non-monotone and this point is past the turn |

Add ±120 if the budget allows: the turnover sits between +80 and +160 and two
points cannot locate it.

## Three findings the grid encodes

**1. The old range was entirely in the flat region.** The first pilot ran
c = 2.0, 4.9, 10.8, chosen to hit alignments of 0.2/0.5/0.8. Every arm came
back at control validity with no property shift (|t| <= 1.2). Nothing was
wrong with the hook: 100% of molecules changed in every arm and rho and cos
hit their targets to three decimals. The coefficients were simply too small.
The useful range starts at |c| ~ 20, an order of magnitude higher.

**2. The degradation is signed, and the negative side is destructive without
being effective.** At |c| = 40 the positive arm holds 88% validity while the
negative arm falls to 52%, and no negative arm produces any property shift
(all |t| < 1). So the direction appears to work in one sense only, if at all.

**3. Section 3.7's predicted norm minimum is negligible at this site.** The
spec says the activation norm falls to a minimum and then grows. It does, but
the unsteered activation is nearly orthogonal to the steering direction
(|cos| < 0.06 at every timestep measured), so the minimum sits at |c| < 0.43
and is less than 1% deep. rho is very nearly an even function of c: 2.75 at
+20 against 2.81 at -20. The signed asymmetry that does exist is in the
RESPONSE, not the norm. Signed tabulation is still required, but the stated
reason for it does not hold here and the text should say so.

## The caveat that governs the sweep

**The apparent property shift is molecule size.** The proxy is Crippen molar
refractivity, which is physically tied to polarisability but scales with
size. Three readings of the same paired data:

| c | dMolMR | t | dMR per heavy atom | t | dMolMR adjusted for dHeavy | t | dHeavy |
|---|---|---|---|---|---|---|---|
| +40 | +0.572 | 2.6 | -0.054 | -1.6 | +0.219 | 1.2 | +0.318 |
| +80 | +0.735 | 2.5 | -0.069 | -1.7 | +0.242 | 0.9 | +0.405 |
| +160 | +0.385 | 2.0 | -0.048 | -1.6 | +0.140 | 0.8 | +0.243 |

Only the raw column is significant. Normalised per heavy atom it reverses
sign; adjusted for dHeavy as a covariate it drops to t ~ 1. Steering is
making molecules bigger, and MolMR follows.

Consequences for the sweep:

- The primary endpoint must be **size-adjusted**, not raw. Report dMolMR,
  dMolMR per heavy atom and dHeavy together, exactly as above.
- MolMR is a graph descriptor and a proxy. The sweep needs an independent
  physical estimate: GFN2-xTB, about a second per molecule, so 4,400
  molecules is roughly 75 minutes. Check it is installed before launching.
- n = 50 is too small. Standard errors on dMolMR were 0.22 to 0.38 against
  effects of 0.2 to 0.7. At n = 400 they fall to about 0.08 and the turnover
  between +80 and +160 becomes resolvable.
- The one candidate size-independent signal, dMR per heavy atom = +0.104
  (t = 3.2) at c = +20, is a single result among 24 comparisons at n = 44.
  It is worth replicating at n = 400, not worth believing yet.

## Cost

11 arms x 400 molecules = 4,400 molecules, about 4.2 h at the measured
1,046 mol/h, plus ~75 min if xTB scores every molecule.

## Command

    cd <repo>/midi
    python3.11 -u ..\scripts\run_pilot.py \
      dataset=qm9 dataset.remove_h=False +experiment=qm9_with_h_adaptive \
      general.wandb=disabled general.gpus=1 \
      general.test_only=<abs path to checkpoint> \
      "+pilot.coeffs=[-160,-80,-40,-20,-10,10,20,40,80,160]" \
      +pilot.n=400 +pilot.window=late +pilot.direction=dom \
      +pilot.store=<abs>/activations_test_full \
      +pilot.dirs=<abs>/results/directions \
      +pilot.out=<abs>/results/sweep

Resumable: completed arms are skipped, chunks are written atomically.
