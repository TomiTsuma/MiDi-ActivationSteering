# Degradation metrics and admissibility tolerances

Written 2026-11-15, before any baseline run exists. Fixes the metric
definitions, the tolerance rule and the admissibility test. Every number in
this document is a rule or a multiplier. No measured value appears, because
none has been computed yet.

## Disclosure, stated first because it limits what this document can claim

The main sweep (`results/sweep`, 23 arms) was generated on 14 to 15 November
and analysed on 15 November, and its results were reported before this
document was written. See `results/sweep_findings.md`, committed as 7203ef4.

So this is **not** a pre-registration of the degradation tolerances. The
ordering the protocol intended cannot be recovered by backdating.

What is salvaged, and what it is worth:

- The **minimum shift of interest of 0.5 standardised units** was specified in
  the research protocol before any sweep was run. It is the one threshold
  here that genuinely predates the result, and it is applied unchanged.
- The **tolerance rule** below is fixed by stated principle rather than by
  inspection: the multiplier comes from the t distribution at the chosen
  number of seeds, and the standardisation denominators come from the
  unsteered baseline runs, which carry no information about which steered arm
  wins. Committing this file before those runs exist is the audit trail.
- The manuscript must say, in the methods, that the sweep was analysed before
  the tolerance table was constructed. That is a disclosed limitation.

Any later change to a threshold in this document must be a new commit with a
reason, not an edit, so the history shows whether a threshold moved after a
result was seen.

## 1. Metric definitions

All five are computed from the stored graph records (`atom_types`, `charges`,
`bond_types`, `positions`) in the `chunk_*.pt` files, so no regeneration is
needed and every set already on disk is evaluable.

**Validity.** An RDKit molecule is built from the graph and passed to
`Chem.SanitizeMol` with default flags. Valid is "did not raise". The
denominator is **every generated molecule in the set**, not the subset that
built successfully. The module asserts agreement with `_count_valid` in
`midi/generate.py`, because `summary.csv` was written by that function and
the two must not diverge.

**Uniqueness.** Among valid molecules, the fraction whose canonical SMILES is
distinct: `|set(canonical SMILES)| / |valid|`. Canonicalisation is
`Chem.MolToSmiles` with default settings, isomeric SMILES off, so
stereochemistry is not part of identity.

**Novelty.** Among the distinct canonical SMILES, the fraction absent from the
checkpoint's own training partition, obtained as
`datamodule.train_dataloader().dataset.smiles` and canonicalised by the same
call. Only the training partition is meaningful here, since the checkpoint
was not trained in this work.

**Molecular stability.** MiDi's own definition, by calling
`midi/analysis/rdkit_functions.py::check_stability` (line 89, with
`allowed_bonds` at line 8). Not reimplemented. Molecule stability is the
fraction of molecules in which every atom carries an allowed valency; atom
stability is the fraction of atoms that do. Both are reported.

**Frechet ChemNet Distance.** `fcd_torch` between the valid canonical SMILES
of the set under test and those of an unsteered reference set. Reported
additionally against the training partition, which is the more usual
quantity, but admissibility is judged on steered against unsteered as the
protocol specifies.

## 2. The sample-size rule, which is not optional

FCD, uniqueness and novelty all depend on how many molecules go into them.
FCD is biased upward at small n; uniqueness and novelty rise as the
denominator falls. Valid counts across the sweep's arms differ by roughly a
factor of two, so comparing arms at their own n would manufacture
differences out of validity alone.

Therefore:

- **Validity, molecule stability and atom stability** are reported at the
  full generated set size, whose denominator is fixed at 500 by design and is
  identical in every arm. No matching needed.
- **Uniqueness, novelty and FCD** are evaluated on a random subsample of
  exactly **m = 250** valid molecules, drawn without replacement, repeated
  **R = 200** times, reporting the mean and the standard deviation across
  repeats. Both sides of an FCD are subsampled to m.
- m = 250 is half the generated set size, fixed a priori so that any arm with
  validity of 50% or better is evaluable. **An arm with fewer than 250 valid
  molecules is reported as not evaluable on these three metrics**, and is not
  compared at a smaller n. Reporting it at its own n would be the exact
  artefact this rule exists to prevent.
- The subsample RNG is seeded from the arm name, so the figure reproduces.

## 3. The unsteered baseline

Ten runs, 500 molecules each, no hooks attached, run seeds

    90001 90002 90003 90004 90005 90006 90007 90008 90009 90010

Ten rather than three. With S = 3 the between-run standard deviation carries
two degrees of freedom, and a tolerance interval for a single new run is then
`t(0.975, 2) * sqrt(1 + 1/3) = 4.97` standard deviations wide, which is not
the "2 sd" the protocol imagined. At S = 10 the same interval is 2.37
standard deviations. The cost is about 3.3 hours at the measured unsteered
rate, which is affordable and buys a defensible table.

The sweep's own `control` arm (run seed 1234) is reported alongside the
baseline as a consistency check. It is **not** included in the baseline mean
or standard deviation, because it is the paired reference for every steered
arm and must stay independent of the tolerance it is judged against.

## 4. The tolerance rule

For each metric, let `mu` be the mean across the S = 10 baseline runs and `s`
the sample standard deviation across them.

**Count metrics** (validity, uniqueness, novelty, molecule stability, atom
stability) also have an analytic sampling error. For an observed proportion
`p` on `n` items, `sigma_bin = sqrt(p * (1 - p) / n)`. The scale used is

    sigma_eff = max(s, sigma_bin)

so a baseline that happens to scatter less than sampling theory allows cannot
produce an absurdly tight tolerance. Comparing `s` against `sigma_bin` is
also a real test of whether the generator is stationary across seeds, and
that comparison is reported.

**The tolerance band** for a single new run is

    mu  +-  k * sigma_eff ,    k = t(0.975, S - 1) * sqrt(1 + 1/S)
                                 = 2.262 * 1.0488
                                 = 2.372

k is fixed at **2.372** by this document and is not to be retuned.

**FCD has no baseline mean to sit around**, because the quantity of interest
is a distance that is zero only for identical sets. Its null is the
distribution of FCD between two *independent unsteered* runs at matched n.
With ten baseline runs there are 45 such pairs, each subsampled R times. The
tolerance is the **upper 97.5th percentile** of that pooled null
distribution. A steered set is within tolerance if its FCD against the
unsteered reference falls below that percentile.

**All bands are two-sided** for the count metrics. Validity and stability
falling is obvious degradation, but novelty or uniqueness rising is also a
sign the output distribution has moved, so neither tail is ignored.

## 5. Admissibility

A configuration is **admissible** only if every one of validity, uniqueness,
novelty, molecule stability and atom stability lies inside its tolerance
band, and its FCD lies below the FCD percentile. An arm that is not evaluable
on uniqueness, novelty or FCD under the m = 250 rule is **inadmissible by
default**, since the reason it is not evaluable is that fewer than half its
molecules are valid.

Admissibility is a gate, not a result. A property shift is reportable only
from an admissible configuration. The central table is shift beside
degradation, in that order, and never shift alone.

## 6. The minimum shift of interest

A property shift is of interest only at **0.5 standardised units or more**.
This threshold predates the sweep and is applied unchanged.

The standardisation denominator is the **per-molecule standard deviation of
the endpoint, pooled across the ten unsteered baseline runs**. Not the
sweep's control arm, so the denominator is independent of the sweep, and not
the standard deviation of the paired difference, which is smaller and would
inflate the standardised effect. This follows the usual convention in which a
standardised mean difference is scaled by the population spread.

The paired-difference standardisation is reported alongside, labelled as
secondary, because the two differ materially and a reader must see which is
being quoted.

The endpoints to which this applies are the three already in use: MolMR, MolMR
per heavy atom, and heavy-atom count, each as a paired difference from the
matched-seed unsteered control.

MolMR remains an RDKit graph descriptor and a proxy for polarisability
through the Lorentz-Lorenz relation, not a measurement. An independent
estimate with GFN2-xTB can be computed over the same saved chunks and is
still outstanding. The minimum shift of interest applies to whichever
endpoint is being quoted, proxy or not.

## 7. Versions to pin in every run manifest

python, numpy, torch, rdkit, fcd_torch, the repository commit, and the
SHA256 of the checkpoint file. Recorded by the metrics module itself, not by
hand.

## 8. What this protocol will conclude about the current sweep

Stated here so that it cannot later look like a finding that emerged from a
threshold chosen to produce it. From `results/sweep_findings.md`, the largest
real-arm effects are MolMR +0.565 and MolMR per heavy atom +0.1032. Against a
control-arm per-molecule spread of roughly 3.47 and 0.35 respectively, those
are about 0.16 and 0.30 standardised units. Both fall short of 0.5.

The expectation, therefore, is that **no arm in the current sweep produces an
admissible property shift of interest**, including the c = +20 arm that
survived the Section 3.8 controls. The baseline runs will set the degradation
tolerances properly and may change which arms are admissible at all, but
they cannot raise an effect of 0.30 standardised units to 0.5.

If the measured outcome differs from this expectation, that difference is
itself worth reporting, and this paragraph is the record of what was expected
before the baselines were run.
