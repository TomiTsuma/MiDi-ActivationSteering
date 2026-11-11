#!/usr/bin/env bash
#
# Phase One end to end: the Block 1 determinism work (4 Nov) and the Blocks 3-6
# capture work (5 Nov), in dependency order, with gates.
#
#   bash scripts/run_phase1.sh                 # everything, in order
#   bash scripts/run_phase1.sh --list          # show the stages
#   bash scripts/run_phase1.sh --from 06_pilot # resume from a stage
#   bash scripts/run_phase1.sh --only 04_liveness
#   SKIP_BASELINE=1 bash scripts/run_phase1.sh # stop before the weekend run
#
# Windows: run under Git Bash. Paths handed to Python are converted to
# C:/... form with cygpath, because the Python interpreter is Windows-native
# and does not understand /c/... paths.
#
# Gates (a failure stops the script):
#   01_unit          the tape and the categorical substitution
#   02_t3_cpu        per-molecule reproducibility, bit-exact. Without this the
#                    Phase Two arms do not pair and the capture is a throwaway.
#   07_verify_pilot  store structure, label alignment, sanity probe
#   09_verify_full   the same on the full store
#
# Non-gating but recorded: 03_t3_gpu, 04_liveness, 05_rotation.

set -euo pipefail

# ---------------------------------------------------------------- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then
  win() { cygpath -m "$1"; }
else
  win() { printf '%s' "$1"; }
fi
REPO_WIN="$(win "$REPO")"

PY="python3.11"
CKPT_WIN="${CKPT:-$REPO_WIN/midi/checkpoints/checkpoint_qm9_h_adaptive.ckpt}"
RUN_SEED="${RUN_SEED:-1234}"
N_PILOT="${N_PILOT:-2000}"
N_FULL="${N_FULL:-13055}"          # the whole QM9 test split after filtering
EVERY="${EVERY:-50}"               # capture every 50th of the 500 steps -> 10
BATCH="${BATCH:-32}"
BASELINE_N="${BASELINE_N:-10000}"  # generated molecules for the Table 2 comparison
T3_GATE_GPUS="${T3_GATE_GPUS:-1}"  # device for the T3 gate: 1 = GPU, 0 = CPU
T3_POS_TOL="${T3_POS_TOL:-1e-3}"   # Angstrom; positions drift ~2e-5 across batch shapes
GEN_CHUNK="${GEN_CHUNK:-250}"      # molecules per checkpointed chunk
GEN_OUT="${GEN_OUT:-$REPO_WIN/baseline_run}"  # chunk store; resume reads this
STORE_PILOT_WIN="$REPO_WIN/activations_pilot"
STORE_FULL_WIN="$REPO_WIN/activations_test_full"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="$REPO/logs/phase1-$STAMP"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/COMPLETION.md"

BASE_OVERRIDES=(
  dataset=qm9
  dataset.remove_h=False
  +experiment=qm9_with_h_adaptive
  general.wandb=disabled
)

STAGES=(00_preflight 01_unit 02_t3_gate 03_t3_crosscheck 04_liveness 05_rotation
        06_pilot 07_verify_pilot 08_full 09_verify_full 10_baseline 11_summary)

# ---------------------------------------------------------------- utils ----
say()  { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }
note() { printf '     %s\n' "$*"; }
die()  { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

run_logged() {           # run_logged <stage> <cmd...>
  local stage="$1"; shift
  local log="$LOGDIR/$stage.log"
  note "log: $log"
  if "$@" 2>&1 | tee "$log"; then
    return 0
  else
    return "${PIPESTATUS[0]}"
  fi
}

# ------------------------------------------------------------- stages ------
s_00_preflight() {
  say "00 preflight"
  note "repo        $REPO"
  note "repo (win)  $REPO_WIN"
  command -v "$PY" >/dev/null || die "python not on PATH. conda activate midi first."
  note "python      $("$PY" -c 'import sys;print(sys.version.split()[0])')"
  "$PY" - <<'EOF' || exit 1
import sys
try:
    import torch, torch_geometric, rdkit, numpy, hydra
except Exception as e:
    print("missing dependency:", e); sys.exit(1)
print(f"     torch       {torch.__version__}  cuda={torch.version.cuda}  "
      f"available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"     gpu         {torch.cuda.get_device_name(0)}")
print(f"     numpy       {numpy.__version__}")
EOF
  "$PY" -c "import midi, midi.rng, midi.capture, midi.verify_store, midi.qm9_labels" \
    || die "midi package not importable. Run: pip install -e . from $REPO"
  note "midi package importable, including rng/capture/verify_store/qm9_labels"

  [ -f "$REPO/midi/checkpoints/checkpoint_qm9_h_adaptive.ckpt" ] \
    || die "checkpoint missing: midi/checkpoints/checkpoint_qm9_h_adaptive.ckpt"
  for f in train_h.pt val_h.pt test_h.pt; do
    [ -f "$REPO/data/qm9/qm9_pyg/processed/$f" ] \
      || die "processed split missing: data/qm9/qm9_pyg/processed/$f"
  done
  for f in uncharacterized.txt test.csv gdb9.sdf; do
    [ -f "$REPO/data/qm9/qm9_pyg/raw/$f" ] \
      || die "raw file missing: data/qm9/qm9_pyg/raw/$f (needed for DFT labels)"
  done
  note "checkpoint, processed splits and raw label files all present"

  # the full store is 8.22 GiB; pilot 1.26 GiB; ask for 12 GiB of headroom
  local avail_kb
  avail_kb="$(df -Pk "$REPO" | awk 'NR==2{print $4}')"
  local avail_gib=$(( avail_kb / 1024 / 1024 ))
  note "free disk    ${avail_gib} GiB (need ~12 GiB for both stores)"
  [ "$avail_gib" -ge 12 ] || die "under 12 GiB free; the full store alone is 8.22 GiB"
}

s_01_unit() {
  say "01 unit tests: tape, keying, categorical substitution  [GATE]"
  cd "$REPO"
  run_logged 01_unit_determinism "$PY" -u tests/test_determinism.py \
    || die "tests/test_determinism.py"
  run_logged 01_unit_discrete "$PY" -u tests/test_discrete_features.py \
    || die "tests/test_discrete_features.py"
}

s_02_t3_gate() {
  say "02 T3 gate: molecule i is batch-independent  [GATE]  (gpus=$T3_GATE_GPUS)"
  note "the gate runs on the device the capture will use, so the reproducibility"
  note "being certified is the one that actually matters"
  note "discrete channels must be bit-exact; positions are allowed $T3_POS_TOL Angstrom"
  cd "$REPO/midi"
  run_logged 02_t3_gate "$PY" -u test_pairing.py "${BASE_OVERRIDES[@]}" \
      "general.gpus=$T3_GATE_GPUS" "general.test_only=$CKPT_WIN" \
      "+pairing.pos_tol=$T3_POS_TOL" \
    || die "T3 failed. Read $LOGDIR/02_t3_gate.log: the verdict names which of the
        three checks broke. A discrete channel changing with batch composition is
        the real pairing bug; a position deviation alone is not."
  grep -q "T3 PASS" "$LOGDIR/02_t3_gate.log" \
    || die "T3 did not print PASS; see $LOGDIR/02_t3_gate.log"
}

s_03_t3_crosscheck() {
  local other=0
  [ "$T3_GATE_GPUS" = "0" ] && other=1
  say "03 T3 cross-check on the other device (gpus=$other, non-gating)"
  note "confirms the verdict is not device-specific"
  cd "$REPO/midi"
  if run_logged 03_t3_crosscheck "$PY" -u test_pairing.py "${BASE_OVERRIDES[@]}" \
      "general.gpus=$other" "general.test_only=$CKPT_WIN" \
      "+pairing.pos_tol=$T3_POS_TOL"; then
    note "cross-check passed too"
  else
    note "cross-check did not pass. The gate already passed on the device that"
    note "matters, so this is recorded and not fatal. See"
    note "$LOGDIR/03_t3_crosscheck.log"
  fi
}

s_04_liveness() {
  say "04 liveness screen: which (layer, channel) sites can be steered at all"
  note "expect y at layer 11 to report DEAD (mlp_out_y is commented out)"
  cd "$REPO/midi"
  run_logged 04_liveness "$PY" -u capture.py "${BASE_OVERRIDES[@]}" \
      general.gpus=1 "general.test_only=$CKPT_WIN" \
      +capture.mode=liveness +capture.liveness_n=8 +capture.liveness_t=400 \
      +capture.liveness_alpha=4.0 "+capture.store=$STORE_PILOT_WIN" \
    || note "liveness screen failed; recorded, not gating"
}

s_05_rotation() {
  say "05 rotation invariance of the scalar capture sites (Section 3.6)"
  cd "$REPO/midi"
  run_logged 05_rotation "$PY" -u capture.py "${BASE_OVERRIDES[@]}" \
      general.gpus=1 "general.test_only=$CKPT_WIN" \
      +capture.mode=rotation +capture.liveness_n=8 +capture.liveness_t=400 \
      "+capture.store=$STORE_PILOT_WIN" \
    || note "rotation check failed; recorded, not gating"
}

s_06_pilot() {
  say "06 pilot capture: $N_PILOT molecules, every=$EVERY  (~1.26 GiB)"
  cd "$REPO/midi"
  run_logged 06_pilot "$PY" -u capture.py "${BASE_OVERRIDES[@]}" \
      general.gpus=1 "general.test_only=$CKPT_WIN" \
      +capture.mode=encode +capture.split=test \
      "+capture.n_molecules=$N_PILOT" "+capture.every=$EVERY" \
      "+capture.batch_size=$BATCH" "+capture.run_seed=$RUN_SEED" \
      "+capture.store=$STORE_PILOT_WIN" \
    || die "pilot capture"
}

s_07_verify_pilot() {
  say "07 verify the pilot store  [GATE]"
  cd "$REPO"
  run_logged 07_verify_pilot "$PY" -u -m midi.verify_store --store "$STORE_PILOT_WIN" \
    || die "the pilot store did not verify. Read $LOGDIR/07_verify_pilot.log:
        if rung 1 (atom count) is flat the store is broken, not the science."
}

s_08_full() {
  say "08 full-split capture: $N_FULL molecules  (~8.22 GiB)"
  note "verify_smiles=true replays the SDF once to prove label alignment (~90 s)"
  cd "$REPO/midi"
  run_logged 08_full "$PY" -u capture.py "${BASE_OVERRIDES[@]}" \
      general.gpus=1 "general.test_only=$CKPT_WIN" \
      +capture.mode=encode +capture.split=test \
      "+capture.n_molecules=$N_FULL" "+capture.every=$EVERY" \
      "+capture.batch_size=$BATCH" "+capture.run_seed=$RUN_SEED" \
      +capture.verify_smiles=true "+capture.store=$STORE_FULL_WIN" \
    || die "full-split capture"
}

s_09_verify_full() {
  say "09 verify the full store  [GATE]"
  cd "$REPO"
  run_logged 09_verify_full "$PY" -u -m midi.verify_store --store "$STORE_FULL_WIN" \
    || die "the full store did not verify; see $LOGDIR/09_verify_full.log"
}

s_10_baseline() {
  if [ "${SKIP_BASELINE:-0}" = "1" ]; then
    say "10 baseline metrics run  [SKIPPED, SKIP_BASELINE=1]"
    return 0
  fi
  say "10 baseline: $BASELINE_N molecules, chunked and resumable"
  note "chunks of $GEN_CHUNK, each written atomically before the next starts, so a"
  note "kill costs at most one chunk. Progress: $GEN_OUT/progress.csv"
  note "re-running this stage RESUMES from the chunks already on disk"
  note "a 2026-11-09 run of 10,000 lost 31h to a system shutdown because the old"
  note "path held everything in memory and wrote nothing until the end"
  cd "$REPO/midi"
  run_logged 10_baseline "$PY" -u generate.py "${BASE_OVERRIDES[@]}" \
      general.gpus=1 "general.test_only=$CKPT_WIN" \
      "+gen.target=$BASELINE_N" "+gen.chunk=$GEN_CHUNK" \
      "+gen.out=$GEN_OUT" "+gen.run_seed=$RUN_SEED" \
    || note "baseline exited non-zero. Completed chunks are intact in $GEN_OUT;
        re-run 'bash scripts/run_phase1.sh --only 10_baseline' to resume."
}

s_11_summary() {
  say "11 completion summary"
  LOGDIR="$LOGDIR" SUMMARY="$SUMMARY" \
  STORE_PILOT="$STORE_PILOT_WIN" STORE_FULL="$STORE_FULL_WIN" \
  REPO_ENV="$REPO" "$PY" - <<'EOF'
import json, os, re

logdir  = os.environ["LOGDIR"]
summary = os.environ["SUMMARY"]
stores  = {"pilot": os.environ["STORE_PILOT"], "full": os.environ["STORE_FULL"]}

def manifest(p):
    try:
        with open(os.path.join(p, "manifest.json")) as f: return json.load(f)
    except Exception: return None

def logtail(name, n=4):
    p = os.path.join(logdir, name)
    if not os.path.exists(p): return "(not run)"
    lines = [l.rstrip() for l in open(p, errors="replace") if l.strip()]
    return "\n".join(lines[-n:]) if lines else "(empty)"

L = []
L.append("# Phase One completion report\n")
L.append(f"Logs: `{logdir}`\n")

L.append("## Gates\n")
L.append("| stage | result |")
L.append("|---|---|")
for stage, label in [("01_unit_determinism", "tape and keying unit tests"),
                     ("01_unit_discrete", "categorical substitution tests"),
                     ("02_t3_gate", "T3 pairing, gate device (GATE)"),
                     ("03_t3_crosscheck", "T3 pairing, cross-check device"),
                     ("07_verify_pilot", "pilot store verified (GATE)"),
                     ("09_verify_full", "full store verified (GATE)")]:
    p = os.path.join(logdir, stage + ".log")
    if not os.path.exists(p):
        r = "not run"
    else:
        t = open(p, errors="replace").read()
        if "T3 PASS" in t or "VERDICT: store is usable" in t or re.search(r"\d+/\d+ passed", t):
            m = re.search(r"(\d+)/(\d+) passed", t)
            r = f"pass ({m.group(0)})" if m else "pass"
        elif "T3 FAIL" in t or "do not build on this store" in t:
            r = "FAIL"
        else:
            r = "see log"
    L.append(f"| {label} | {r} |")

L.append("\n## Measured sampling rate\n")
for tag, path in stores.items():
    m = manifest(path)
    if not m:
        L.append(f"- {tag}: no manifest at `{path}`"); continue
    L.append(f"- **{tag}**: {m['n_molecules']:,} molecules, "
             f"{m['n_layers']} layers, timesteps {m['timesteps']}, "
             f"**{m.get('molecules_per_gpu_hour', float('nan')):,.0f} mol/GPU-hour**, "
             f"{m['store_bytes']/1024**3:.2f} GiB, "
             f"elapsed {m.get('elapsed_seconds', 0)/60:.1f} min")
    L.append(f"  - torch {m.get('torch')}, cuda {m.get('cuda')}, "
             f"gpu {m.get('gpu_name')}, commit `{m.get('git_commit')}`")
    L.append(f"  - run_seed {m.get('run_seed')}, n_max {m.get('n_max')}, T {m.get('T')}")
L.append("\n> Put this rate into Section 3.11, which currently asserts "
         "472 mol/GPU-hour as a projection rather than a measurement.\n")

lv = os.path.join(stores["pilot"], "liveness.json")
L.append("## Liveness screen\n")
if os.path.exists(lv):
    rows = json.load(open(lv))
    dead = [r for r in rows if r["dead"]]
    verb = "site is" if len(dead) == 1 else "sites are"
    L.append(f"{len(dead)} of {len(rows)} (layer, channel) {verb} causally dead.\n")
    if dead:
        L.append("| layer | channel |")
        L.append("|---|---|")
        for r in dead:
            L.append(f"| {r['layer']} | {r['channel']} |")
    L.append("\n> A dead site can be probed but never steered. Exclude these from "
             "the Phase Two injection sites. `y` at layer 11 is expected here, "
             "because `mlp_out_y` is commented out in `transformer_model.py`.\n")
else:
    L.append("(liveness.json not found)\n")

L.append("## Rotation invariance\n```\n" + logtail("05_rotation.log", 6) + "\n```\n")

L.append("## Baseline metrics versus MiDi published\n")
L.append("Published figures are Table 2 of `MiDi_paper.pdf`, the "
         "**MiDi (adaptive)** row, which is the checkpoint in use.\n")
L.append("| Metric | Published (adaptive) | Yours |")
L.append("|---|---|---|")
for k, v in [("Molecule stable", "97.5 +/- 0.1"), ("Atom stable", "99.8 +/- 0.0"),
             ("Validity", "97.9 +/- 0.1"), ("Uniqueness", "97.6 +/- 0.1"),
             ("Novelty", "67.5 +/- 0.3"), ("Connected", "99.9 +/- 0.0"),
             ("Valency (e-2, lower better)", "0.3"), ("Atom (e-2)", "0.3"),
             ("Bond (e-2)", "0.0"), ("Angles (deg)", "0.62 +/- 0.02"),
             ("Bond lengths (e-2 A)", "0.3 +/- 0.1")]:
    L.append(f"| {k} | {v} | |")
L.append(f"\nFill the right column from `{logdir}/10_baseline.log`. "
         "Do not compare a 32-molecule run against these: uniqueness and novelty "
         "on 32 samples are close to meaningless, and the published figures come "
         "from 10,000.\n")
_m = manifest(stores["full"]) or manifest(stores["pilot"]) or {}
_tv = _m.get("torch", "unknown")
_cv = _m.get("cuda", "unknown")
L.append("If molecule stability or validity lands well below the published row, "
         f"the first suspect is the environment: torch {_tv} / cuda {_cv} here "
         "against torch 2.0.1 / cu118 / Python 3.9 in `requirements.txt` and the "
         "README. Sections 3.11 and 3.12 commit to pinning versions under which "
         "the checkpoint reproduces its published metrics, so this is that "
         "commitment being tested.\n")

L.append("## Still to do\n")
L.append("- record the rate in the run manifest and Section 3.11")
L.append("- record the liveness table as the measured basis for excluding dead sites")
L.append("- commit the Block 1 patch and the Blocks 3-6 files as separate commits, "
         "so the determinism change and the capture hooks stay separable in the "
         "patch Section 3.12 requires")
L.append("- labels for LogP and aqueous solubility are not DFT: they are computed, "
         "and are not in this store yet")

open(summary, "w").write("\n".join(L) + "\n")
print(f"wrote {summary}")
EOF
  note "summary: $SUMMARY"
}

# ----------------------------------------------------------------- main ----
FROM=""; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list) printf '%s\n' "${STAGES[@]}"; exit 0 ;;
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

started=0
printf '\n\033[1mPhase One run %s\033[0m\n' "$STAMP"
printf 'logs -> %s\n' "$LOGDIR"
for st in "${STAGES[@]}"; do
  if [ -n "$ONLY" ]; then [ "$st" = "$ONLY" ] || continue; fi
  if [ -n "$FROM" ] && [ "$started" -eq 0 ]; then
    [ "$st" = "$FROM" ] && started=1 || continue
  fi
  "s_$st"
done

say "done"
note "completion report: $SUMMARY"
