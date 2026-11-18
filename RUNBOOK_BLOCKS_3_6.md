# Blocks 3-6: capture, verify, launch

## Read this first: there is no capture to check

The task as written assumes last night's capture exists. It does not. Block 3
was never implemented, there is no `capture.py` in the repo as of this morning,
and there is no activation store anywhere under the project. So steps 1 to 4 of
the task ("check the store, sanity probe, fix what is broken") have nothing to
operate on yet.

What **did** happen on 4 Nov, confirmed from disk:

| Evidence | Meaning |
|---|---|
| `midi/rng.py`, `_compat.py`, `test_pairing.py` present | Block 1 files in place |
| `diffusion_model.py` 29,931 B, `noise_model.py` 21,976 B, both with `tape`/`mol_ids`/`rng` markers | all five Block 1 edits applied correctly |
| `data/qm9/qm9_pyg/processed/{train,val,test}_h.pt` | all three splits processed with explicit H |
| `outputs/2026-11-04/06-48-21-.../final_smiles_GR0_0.txt`, 32 SMILES | a 32-molecule sampling run succeeded end to end |
| `outputs/2026-11-04/09-53-02-.../` with `test_pairing.log` | T3 was launched on CPU with the right overrides |
| `docs/changelog/*` x2 | the Windows NCCL/DDP fix and a Matplotlib `w_xaxis` fix |

So Block 1 is done and the baseline sampler works. **Confirm the T3 verdict
before using any of this**, because the hydra log is empty (0 bytes, as it is
for every run here) and the pass/fail was only printed to the console.

## Second correction: the sizing is inverted

The weekend plan assumes the capture is the big job. It is not, by a factor of
about fifty.

* generation costs **500** denoiser forward passes per molecule, one per reverse step
* encode capture costs **10**, one per captured timestep at `every=50`

| N molecules | every | timesteps | forward passes | store |
|---|---|---|---|---|
| 2,000 | 50 | 10 | 20,000 | 1.26 GiB |
| 5,000 | 50 | 10 | 50,000 | 3.15 GiB |
| **13,055 (whole test split)** | **50** | **10** | **130,550** | **8.22 GiB** |
| 13,055 | 25 | 20 | 261,100 | 16.43 GiB |

At any plausible rate the full-split capture finishes in **under three hours**,
and the 8.22 GiB store lands inside the 10 GB headroom. There is no reason to
settle for 2,000 molecules.

What actually needs the weekend is step 3, the baseline metrics run: 10,000
**generated** molecules at 500 steps each is 5,000,000 forward passes.

| generation rate | 10k baseline | full-split capture |
|---|---|---|
| 100 mol/GPU-h | 100 h | 2.6 h |
| 200 mol/GPU-h | 50 h | 1.3 h |
| 400 mol/GPU-h | 25 h | 0.7 h |
| 800 mol/GPU-h | 12.5 h | 0.3 h |

If the measured rate is under about 200 mol/GPU-h, 10,000 molecules will not
fit in a weekend on one card. Split 5,000 per card across both, or drop to
5,000 total and say so. Do not drop below 5,000: Section 2.6 of your own review
records that FCD needs at least 5,000 designs per set.

## Third correction: Phase One must be an encode pass, not a generation pass

This one changes the science, not just the schedule.

Five of the seven targets (polarisability, heat capacity, dipole moment, HOMO,
LUMO) are DFT labels. They exist only for real QM9 molecules. A molecule the
model invents during generation has no such label, and getting one would mean
running DFT on every sample. So RQ1 is answerable only by taking **labelled**
molecules, noising them to a chosen timestep, passing them through the frozen
denoiser, and probing those activations.

`capture.py` therefore defaults to `mode=encode`. Generation-time capture is
what Phase Two needs; it is not what Phase One can probe.

## Fourth correction: the labels are not in the dataset, and the join has a trap

`QM9Dataset.process()` reads the DFT label CSV **only as an index filter** and
never attaches the labels to the stored molecules. They live only in
`data/qm9/qm9_pyg/raw/`.

The trap: `download()` writes each split CSV from a shuffled frame
(`df.sample(frac=1, random_state=42)`), but `process()` iterates the SDF in
increasing order. So dataset position *j* is the *j*-th smallest surviving SDF
index, **not** CSV row *j*. Joining on CSV row order mislabels every molecule,
and the symptom is a flat probe that reads as a negative scientific result.

Verified against your actual files for the test split: 13,388 CSV rows, 333
dropped as uncharacterized (3,054 across the whole database), **13,055
survivors**; CSV index *i* carries `mol_id gdb_{i+1}`, so index *i* is SDF
molecule *i*, 0-based. `midi/qm9_labels.py` implements exactly this and asserts
the `mol_id` relation rather than trusting it.

## New files

| Path | Role |
|---|---|
| `midi/qm9_labels.py` | DFT label alignment, with an optional SMILES-replay proof |
| `midi/capture.py` | Blocks 3-5: `mode=encode`, `mode=liveness`, `mode=rotation` |
| `midi/verify_store.py` | the task's steps 1 and 2: structure, alignment, sanity probe |

No edits to existing files. Everything is additive on top of the Block 1 patch.

## Run order for today

All commands from `MiDi-ActivationSteering\midi`, with:

```
$CKPT = "C:/Users/tsuma.thomas/Documents/MiDi-ActivationSteering/midi/checkpoints/checkpoint_qm9_h_adaptive.ckpt"
$BASE = "dataset=qm9 dataset.remove_h=False +experiment=qm9_with_h_adaptive general.wandb=disabled"
```

### 0. Confirm T3 (5 min, do not skip)

```powershell
python test_pairing.py $BASE general.gpus=0 "general.test_only=$CKPT"
```

Must print `T3 PASS`. Everything below assumes per-molecule reproducibility.

### 1. Liveness screen (10 min, GPU)

Cheap, and it settles a question that would otherwise waste the sweep. It
injects a unit random direction at each of the 36 (layer, channel) sites and
reports whether the denoiser's output changes at all.

```powershell
python capture.py $BASE general.gpus=1 "general.test_only=$CKPT" `
  +capture.mode=liveness +capture.liveness_n=8 +capture.liveness_t=400 `
  +capture.store=../activations_pilot
```

**Expected: `y` at layer 11 reports DEAD**, because `mlp_out_y` is commented out
in `transformer_model.py` and `y = y_to_out` passes the input `y` through. If it
reports live, my reading of that file is wrong and the Phase Two site list needs
revisiting. Either way this becomes a measured result you can cite rather than a
code-reading claim.

### 2. Rotation invariance (5 min, GPU)

Section 3.6's own validation, which no run has yet performed.

```powershell
python capture.py $BASE general.gpus=1 "general.test_only=$CKPT" `
  +capture.mode=rotation +capture.liveness_n=8 +capture.liveness_t=400 `
  +capture.store=../activations_pilot
```

Expect deviations around 1e-4, not exact zero: these are float32 forward passes
and rounding accumulates. Anything at 1e-2 or above means the capture site is
not invariant and the probe would be reading orientation rather than structure.

### 3. Pilot capture, 2,000 molecules (minutes, GPU)

```powershell
python capture.py $BASE general.gpus=1 "general.test_only=$CKPT" `
  +capture.mode=encode +capture.split=test +capture.n_molecules=2000 `
  +capture.every=50 +capture.batch_size=32 +capture.run_seed=1234 `
  +capture.store=../activations_pilot
```

Prints the measured rate and writes `manifest.json`. First run also prints the
label alignment summary; it should say 13,388 CSV rows, 333 dropped, 13,055
survive.

Add `+capture.verify_smiles=true` once, ever, to prove the alignment by
replaying the SDF and comparing each molecule's canonical SMILES against the
dataset's own stored `smiles`. It costs about 90 seconds and it is the only
check that can catch an off-by-one.

### 4. Verify the pilot (1 min, no GPU)

```powershell
cd ..
python -m midi.verify_store --store activations_pilot
```

Exits 0 only if the store is usable. The probe is a three-rung ladder, because
one R-squared cannot tell a broken store from a negative result:

| rung | what | expectation |
|---|---|---|
| 1 | atom count from activations | near 1.0. Trivially present in any graph representation. **If this is flat, the store is broken, not the science.** |
| 2 | polarisability from atom count alone | high, about 0.95. Pure chemistry, no activations. Checks the labels independently. |
| 3 | polarisability from activations | only meaningful once 1 and 2 pass |

I validated this discrimination on three synthetic stores with known ground
truth: a sound store reported rung 1 R-squared 0.974 and rung 3 0.952 against a
0.972 size ceiling and exited 0; a store with deliberately shuffled labels
collapsed to −0.35 and was correctly blamed on the store, exiting 1; an all-zero
store was caught structurally before the probe ran, exiting 1.

### 5. Full-split capture (under 3 h, GPU)

Only after step 4 exits 0.

```powershell
cd midi
python capture.py $BASE general.gpus=1 "general.test_only=$CKPT" `
  +capture.mode=encode +capture.split=test +capture.n_molecules=13055 `
  +capture.every=50 +capture.batch_size=32 +capture.run_seed=1234 `
  +capture.verify_smiles=true +capture.store=../activations_test_full
```

8.22 GiB. Verify it the same way before relying on it.

### 6. Launch the weekend baseline run (the thing that needs the weekend)

```powershell
python main.py $BASE general.gpus=1 general.num_final_sampling=1 `
  general.final_model_samples_to_generate=10000 `
  general.final_model_samples_to_save=10 general.final_model_chains_to_save=0 `
  "general.test_only=$CKPT"
```

Then record your figures beside MiDi's published ones. These are Table 2 of
`MiDi_paper.pdf`, the **MiDi (adaptive)** row, which is the checkpoint you have:

| Metric | Published (adaptive) | Yours |
|---|---|---|
| Molecule stable | 97.5 ± 0.1 | |
| Atom stable | 99.8 ± 0.0 | |
| Validity | 97.9 ± 0.1 | |
| Uniqueness | 97.6 ± 0.1 | |
| Novelty | 67.5 ± 0.3 | |
| Connected | 99.9 ± 0.0 | |
| Valency (e-2, lower better) | 0.3 | |
| Atom (e-2) | 0.3 | |
| Bond (e-2) | 0.0 | |
| Angles (deg) | 0.62 ± 0.02 | |
| Bond lengths (e-2 A) | 0.3 ± 0.1 | |

**Do not compare the 32-molecule smoke run against these.** Uniqueness and
novelty on 32 samples are nearly meaningless, and the published figures come
from 10,000 samples. That is why step 6 generates 10,000.

If molecule stability or validity lands well below the published row, the
primary suspect is the environment: you are on torch 2.15 and Python 3.11 where
`requirements.txt` pins torch 2.0.1 and the README specifies Python 3.9 with
cu118. Sections 3.11 and 3.12 commit you to pinning "to those under which the
released checkpoint reproduces its published sampling metrics", so this
comparison is that commitment being tested.

## Record in the manifest and in the dissertation

* the measured rate, into Section 3.11, which currently asserts 472
  mol/GPU-hour as a projection rather than a measurement
* the T3 verdict, plus the CUDA max-deviation figure if the GPU run is not
  bit-exact
* the liveness table, which is the measured basis for excluding `y` at layer 11
  from the Phase Two injection sites
* the rotation-invariance worst deviation, which is Section 3.6's validation
* your Table 2 row beside the published one
