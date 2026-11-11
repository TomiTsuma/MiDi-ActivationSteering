<#
.SYNOPSIS
    Phase One end to end: the Block 1 determinism work and the Blocks 3-6
    capture work, in dependency order, with gates.

.DESCRIPTION
    PowerShell port of scripts/run_phase1.sh. Same twelve stages, same gates,
    same logs and completion report. Use this one on Windows; it needs no
    Git Bash and no cygpath.

    Gates (a failure stops the script):
      01_unit          the tape and the categorical substitution
      02_t3_cpu        per-molecule reproducibility, bit-exact. Without this
                       the Phase Two arms do not pair and the capture is a
                       throwaway.
      07_verify_pilot  store structure, label alignment, sanity probe
      09_verify_full   the same on the full store

    Non-gating but recorded: 03_t3_gpu, 04_liveness, 05_rotation.

.EXAMPLE
    .\scripts\run_phase1.ps1
    Runs every stage in order.

.EXAMPLE
    .\scripts\run_phase1.ps1 -SkipBaseline
    Stages 0 to 9 only. This is the sensible first invocation: it runs
    everything that gates and leaves the long generation run for later.

.EXAMPLE
    .\scripts\run_phase1.ps1 -From 06_pilot
    .\scripts\run_phase1.ps1 -Only 04_liveness
    .\scripts\run_phase1.ps1 -List

.NOTES
    Activate the conda environment first; this script does not do it for you:
        conda activate midi
    If the execution policy blocks it:
        powershell -ExecutionPolicy Bypass -File .\scripts\run_phase1.ps1
#>

[CmdletBinding()]
param(
    [string] $Py        = 'python',
    [string] $Ckpt      = '',
    [int]    $RunSeed   = 1234,
    [int]    $NPilot    = 2000,
    [int]    $NFull     = 13055,   # the whole QM9 test split after filtering
    [int]    $Every     = 50,      # capture every 50th of the 500 steps -> 10
    [int]    $Batch     = 32,
    [int]    $BaselineN = 10000,   # generated molecules for the Table 2 comparison
    [int]    $T3GateGpus = 1,      # device for the T3 gate: 1 = GPU, 0 = CPU
    [string] $T3PosTol  = '1e-3',  # Angstrom; positions drift ~2e-5 across batch shapes
    [int]    $GenChunk  = 250,     # molecules per checkpointed chunk
    [string] $GenOut    = '',      # chunk store; default <repo>/baseline_run
    [switch] $SkipBaseline,
    [string] $From      = '',
    [string] $Only      = '',
    [switch] $List
)

# ------------------------------------------------------------------ paths ---
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
$Repo = (Resolve-Path (Join-Path $ScriptDir '..')).Path

# Hydra parses these values; forward slashes avoid any backslash-escape
# question in the override grammar, and Python on Windows accepts them.
$RepoFwd = $Repo -replace '\\', '/'

if (-not $Ckpt) {
    $Ckpt = "$RepoFwd/midi/checkpoints/checkpoint_qm9_h_adaptive.ckpt"
}
if (-not $GenOut) { $GenOut = "$RepoFwd/baseline_run" }
$StorePilot = "$RepoFwd/activations_pilot"
$StoreFull  = "$RepoFwd/activations_test_full"

$Stamp   = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogDir  = Join-Path $Repo "logs\phase1-$Stamp"
$Summary = Join-Path $LogDir 'COMPLETION.md'
# the directory is created in the main section, after -List has had its say,
# so listing the stages leaves nothing behind

$BaseOverrides = @(
    'dataset=qm9',
    'dataset.remove_h=False',
    '+experiment=qm9_with_h_adaptive',
    'general.wandb=disabled'
)

$Stages = @(
    '00_preflight', '01_unit', '02_t3_gate', '03_t3_crosscheck', '04_liveness',
    '05_rotation', '06_pilot', '07_verify_pilot', '08_full', '09_verify_full',
    '10_baseline', '11_summary'
)

# ------------------------------------------------------------------ utils ---
function Say  ([string] $m) { Write-Host ''; Write-Host "==== $m ====" -ForegroundColor Cyan }
function Note ([string] $m) { Write-Host "     $m" }
function Die  ([string] $m) {
    Write-Host ''
    Write-Host "FAILED: $m" -ForegroundColor Red
    exit 1
}

function Invoke-Logged {
    <#
      Runs a native command, tees combined output to a log, returns its exit code.
      $ErrorActionPreference is relaxed around the call because a native command
      writing to stderr would otherwise surface as a NativeCommandError and abort
      before the exit code can be inspected.
    #>
    param(
        [Parameter(Mandatory = $true)] [string]   $Stage,
        [Parameter(Mandatory = $true)] [string]   $Exe,
        [Parameter(Mandatory = $true)] [string[]] $CmdArgs
    )
    $log = Join-Path $LogDir "$Stage.log"
    Note "log: $log"
    $code = 1          # fail closed: a throw must not read as success
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Out-Host sinks Tee-Object's pass-through. Without it the captured
        # output would become this function's return value and every gate
        # would compare an array against 0.
        & $Exe @CmdArgs 2>&1 | Tee-Object -FilePath $log | Out-Host
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    } finally {
        $ErrorActionPreference = $prev
    }
    return $code
}

function Test-LogContains {
    param([string] $Stage, [string] $Pattern)
    $log = Join-Path $LogDir "$Stage.log"
    if (-not (Test-Path $log)) { return $false }
    return [bool](Select-String -Path $log -Pattern $Pattern -SimpleMatch -Quiet)
}

# ----------------------------------------------------------------- stages ---
function Invoke-Stage_00_preflight {
    Say '00 preflight'
    Note "repo        $Repo"
    Note "repo (fwd)  $RepoFwd"

    $pyCmd = Get-Command $Py -ErrorAction SilentlyContinue
    if (-not $pyCmd) { Die "python not on PATH. Run: conda activate midi" }
    $ver = & $Py -c 'import sys;print(sys.version.split()[0])'
    Note "python      $ver  ($($pyCmd.Source))"

    $probe = @'
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
'@
    $probeFile = Join-Path $LogDir '_probe.py'
    Set-Content -Path $probeFile -Value $probe -Encoding ASCII
    & $Py $probeFile
    if ($LASTEXITCODE -ne 0) { Die 'a required dependency is missing' }

    & $Py -c 'import midi, midi.rng, midi.capture, midi.verify_store, midi.qm9_labels'
    if ($LASTEXITCODE -ne 0) {
        Die "midi package not importable. Run: pip install -e . from $Repo"
    }
    Note 'midi package importable, including rng/capture/verify_store/qm9_labels'

    $ckptLocal = Join-Path $Repo 'midi\checkpoints\checkpoint_qm9_h_adaptive.ckpt'
    if (-not (Test-Path $ckptLocal)) {
        Die 'checkpoint missing: midi\checkpoints\checkpoint_qm9_h_adaptive.ckpt'
    }
    foreach ($f in @('train_h.pt', 'val_h.pt', 'test_h.pt')) {
        if (-not (Test-Path (Join-Path $Repo "data\qm9\qm9_pyg\processed\$f"))) {
            Die "processed split missing: data\qm9\qm9_pyg\processed\$f"
        }
    }
    foreach ($f in @('uncharacterized.txt', 'test.csv', 'gdb9.sdf')) {
        if (-not (Test-Path (Join-Path $Repo "data\qm9\qm9_pyg\raw\$f"))) {
            Die "raw file missing: data\qm9\qm9_pyg\raw\$f (needed for DFT labels)"
        }
    }
    Note 'checkpoint, processed splits and raw label files all present'

    # the full store is 8.22 GiB, the pilot 1.26 GiB; ask for 12 GiB of headroom
    try {
        $drive = (Get-Item $Repo).PSDrive.Name
        $free  = (Get-PSDrive -Name $drive).Free
        $freeGiB = [math]::Floor($free / 1GB)
        Note "free disk   $freeGiB GiB on ${drive}: (need ~12 GiB for both stores)"
        if ($freeGiB -lt 12) {
            Die "under 12 GiB free; the full store alone is 8.22 GiB"
        }
    } catch {
        Note 'could not determine free disk space; need ~12 GiB for both stores'
    }
}

function Invoke-Stage_01_unit {
    Say '01 unit tests: tape, keying, categorical substitution  [GATE]'
    Set-Location $Repo
    $c = Invoke-Logged '01_unit_determinism' $Py @('-u', 'tests\test_determinism.py')
    if ($c -ne 0) { Die 'tests\test_determinism.py' }
    $c = Invoke-Logged '01_unit_discrete' $Py @('-u', 'tests\test_discrete_features.py')
    if ($c -ne 0) { Die 'tests\test_discrete_features.py' }
}

function Invoke-Stage_02_t3_gate {
    Say "02 T3 gate: molecule i is batch-independent  [GATE]  (gpus=$T3GateGpus)"
    Note 'the gate runs on the device the capture will use, so the reproducibility'
    Note 'being certified is the one that actually matters'
    Note "discrete channels must be bit-exact; positions are allowed $T3PosTol Angstrom"
    Set-Location (Join-Path $Repo 'midi')
    $a = @('-u', 'test_pairing.py') + $BaseOverrides +
         @("general.gpus=$T3GateGpus", "general.test_only=$Ckpt",
           "+pairing.pos_tol=$T3PosTol")
    $c = Invoke-Logged '02_t3_gate' $Py $a
    if ($c -ne 0) {
        $msg = "T3 failed. Read $LogDir\02_t3_gate.log: the verdict names which of " +
               'the three checks broke. A discrete channel changing with batch ' +
               'composition is the real pairing bug; a position deviation alone is not.'
        Die $msg
    }
    if (-not (Test-LogContains '02_t3_gate' 'T3 PASS')) {
        Die "T3 did not print PASS; see $LogDir\02_t3_gate.log"
    }
}

function Invoke-Stage_03_t3_crosscheck {
    $other = 0
    if ($T3GateGpus -eq 0) { $other = 1 }
    Say "03 T3 cross-check on the other device (gpus=$other, non-gating)"
    Note 'confirms the verdict is not device-specific'
    Set-Location (Join-Path $Repo 'midi')
    $a = @('-u', 'test_pairing.py') + $BaseOverrides +
         @("general.gpus=$other", "general.test_only=$Ckpt",
           "+pairing.pos_tol=$T3PosTol")
    $c = Invoke-Logged '03_t3_crosscheck' $Py $a
    if ($c -eq 0) { Note 'cross-check passed too' }
    else {
        Note 'cross-check did not pass. The gate already passed on the device that'
        Note 'matters, so this is recorded and not fatal. See'
        Note "$LogDir\03_t3_crosscheck.log"
    }
}

function Invoke-Stage_04_liveness {
    Say '04 liveness screen: which (layer, channel) sites can be steered at all'
    Note 'expect y at layer 11 to report DEAD (mlp_out_y is commented out)'
    Set-Location (Join-Path $Repo 'midi')
    $a = @('-u', 'capture.py') + $BaseOverrides + @(
        'general.gpus=1', "general.test_only=$Ckpt",
        '+capture.mode=liveness', '+capture.liveness_n=8',
        '+capture.liveness_t=400', '+capture.liveness_alpha=4.0',
        "+capture.store=$StorePilot")
    $c = Invoke-Logged '04_liveness' $Py $a
    if ($c -ne 0) { Note 'liveness screen failed; recorded, not gating' }
}

function Invoke-Stage_05_rotation {
    Say '05 rotation invariance of the scalar capture sites (Section 3.6)'
    Set-Location (Join-Path $Repo 'midi')
    $a = @('-u', 'capture.py') + $BaseOverrides + @(
        'general.gpus=1', "general.test_only=$Ckpt",
        '+capture.mode=rotation', '+capture.liveness_n=8',
        '+capture.liveness_t=400', "+capture.store=$StorePilot")
    $c = Invoke-Logged '05_rotation' $Py $a
    if ($c -ne 0) { Note 'rotation check failed; recorded, not gating' }
}

function Invoke-Stage_06_pilot {
    Say "06 pilot capture: $NPilot molecules, every=$Every  (~1.26 GiB)"
    Set-Location (Join-Path $Repo 'midi')
    $a = @('-u', 'capture.py') + $BaseOverrides + @(
        'general.gpus=1', "general.test_only=$Ckpt",
        '+capture.mode=encode', '+capture.split=test',
        "+capture.n_molecules=$NPilot", "+capture.every=$Every",
        "+capture.batch_size=$Batch", "+capture.run_seed=$RunSeed",
        "+capture.store=$StorePilot")
    $c = Invoke-Logged '06_pilot' $Py $a
    if ($c -ne 0) { Die 'pilot capture' }
}

function Invoke-Stage_07_verify_pilot {
    Say '07 verify the pilot store  [GATE]'
    Set-Location $Repo
    $c = Invoke-Logged '07_verify_pilot' $Py @('-u', '-m', 'midi.verify_store', '--store', $StorePilot)
    if ($c -ne 0) {
        $msg = "the pilot store did not verify. Read $LogDir\07_verify_pilot.log: " +
               'if rung 1 (atom count) is flat the store is broken, not the science.'
        Die $msg
    }
}

function Invoke-Stage_08_full {
    Say "08 full-split capture: $NFull molecules  (~8.22 GiB)"
    Note 'verify_smiles=true replays the SDF once to prove label alignment (~90 s)'
    Set-Location (Join-Path $Repo 'midi')
    $a = @('-u', 'capture.py') + $BaseOverrides + @(
        'general.gpus=1', "general.test_only=$Ckpt",
        '+capture.mode=encode', '+capture.split=test',
        "+capture.n_molecules=$NFull", "+capture.every=$Every",
        "+capture.batch_size=$Batch", "+capture.run_seed=$RunSeed",
        '+capture.verify_smiles=true', "+capture.store=$StoreFull")
    $c = Invoke-Logged '08_full' $Py $a
    if ($c -ne 0) { Die 'full-split capture' }
}

function Invoke-Stage_09_verify_full {
    Say '09 verify the full store  [GATE]'
    Set-Location $Repo
    $c = Invoke-Logged '09_verify_full' $Py @('-u', '-m', 'midi.verify_store', '--store', $StoreFull)
    if ($c -ne 0) { Die "the full store did not verify; see $LogDir\09_verify_full.log" }
}

function Invoke-Stage_10_baseline {
    if ($SkipBaseline) {
        Say '10 baseline metrics run  [SKIPPED, -SkipBaseline]'
        return
    }
    Say "10 baseline: $BaselineN molecules, chunked and resumable"
    Note "chunks of $GenChunk, each written atomically before the next starts, so a"
    Note "kill costs at most one chunk. Progress: $GenOut/progress.csv"
    Note 're-running this stage RESUMES from the chunks already on disk'
    Note 'a 2026-11-09 run of 10,000 lost 31h to a system shutdown because the old'
    Note 'path held everything in memory and wrote nothing until the end'
    Set-Location (Join-Path $Repo 'midi')
    $a = @('-u', 'generate.py') + $BaseOverrides + @(
        'general.gpus=1', "general.test_only=$Ckpt",
        "+gen.target=$BaselineN", "+gen.chunk=$GenChunk",
        "+gen.out=$GenOut", "+gen.run_seed=$RunSeed")
    $c = Invoke-Logged '10_baseline' $Py $a
    if ($c -ne 0) {
        Note "baseline exited non-zero. Completed chunks are intact in $GenOut;"
        Note "re-run with -Only 10_baseline to resume."
    }
}

function Invoke-Stage_11_summary {
    Say '11 completion summary'
    $src = @'
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
'@
    $srcFile = Join-Path $LogDir '_summary.py'
    Set-Content -Path $srcFile -Value $src -Encoding ASCII
    $env:LOGDIR      = $LogDir
    $env:SUMMARY     = $Summary
    $env:STORE_PILOT = $StorePilot
    $env:STORE_FULL  = $StoreFull
    & $Py $srcFile
    if ($LASTEXITCODE -ne 0) { Note 'summary generation failed' }
    Note "summary: $Summary"
}

# ------------------------------------------------------------------- main ---
if ($List) { $Stages | ForEach-Object { Write-Output $_ }; exit 0 }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

foreach ($s in @($From, $Only)) {
    if ($s -and ($Stages -notcontains $s)) {
        Die "unknown stage '$s'. Run with -List to see the stage names."
    }
}

$Transcript = Join-Path $LogDir 'transcript.log'
$haveTranscript = $false
try {
    Start-Transcript -Path $Transcript -Force | Out-Null
    $haveTranscript = $true
} catch {
    Write-Host 'transcript unavailable; progress will appear on the console only'
}

Write-Host ''
Write-Host "Phase One run $Stamp" -ForegroundColor Cyan
Write-Host "logs -> $LogDir"
if ($haveTranscript) { Write-Host "transcript -> $Transcript" }

try {
    $started = $false
    foreach ($st in $Stages) {
        if ($Only) { if ($st -ne $Only) { continue } }
        if ($From -and -not $started) {
            if ($st -eq $From) { $started = $true } else { continue }
        }
        & "Invoke-Stage_$st"
    }

    Say 'done'
    Note "completion report: $Summary"
} finally {
    if ($haveTranscript) { try { Stop-Transcript | Out-Null } catch { } }
}
