# scripts/run_baselines.ps1 -- the ten unsteered baseline runs.
#
# Stage 4 of results/tolerance_protocol.md. Ten runs of 500 molecules with no
# hooks attached, at the seeds the protocol fixed:
#
#     90001 90002 90003 90004 90005 90006 90007 90008 90009 90010
#
# Ten rather than three because at S = 3 the between-run sd carries two
# degrees of freedom and the tolerance interval for a single new run is
# t(0.975,2)*sqrt(1+1/3) = 4.97 sd wide. At S = 10 it is 2.372 sd.
#
# Each seed gets its own output directory, because run_pilot.py resumes by
# ARM NAME and every one of these arms is called "control". Sharing a
# directory would make runs 2 to 10 skip as already done.
#
# "+pilot.coeffs=[]" is what makes a run unsteered: the explicit-coefficient
# branch is entered, contributes no steered arms, and the plan is the single
# shared control arm. The sweep's own control (run seed 1234) is deliberately
# NOT one of these, since it is the paired reference every steered arm is
# measured against and must stay independent of the tolerance it is judged by.
#
# Resumable at run level: a seed whose summary.csv exists is skipped.
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Ckpt = Join-Path $Repo "midi\checkpoints\checkpoint_qm9_h_adaptive.ckpt"
$Store = Join-Path $Repo "activations_test_full"
$Dirs = Join-Path $Repo "results\directions"

foreach ($s in 90001..90010) {
    $Out = Join-Path $Repo "results\baseline\seed$s"
    if (Test-Path (Join-Path $Out "summary.csv")) {
        Write-Host "[baselines] seed $s already complete, skipping"
        continue
    }
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    Write-Host "[baselines] seed $s -> $Out"
    Set-Location (Join-Path $Repo "midi")
    python3.11 -u ..\scripts\run_pilot.py `
      dataset=qm9 dataset.remove_h=False +experiment=qm9_with_h_adaptive `
      general.wandb=disabled general.gpus=1 `
      "general.test_only=$Ckpt" `
      "+pilot.coeffs=[]" `
      +pilot.n=500 "+pilot.run_seed=$s" `
      "+pilot.store=$Store" `
      "+pilot.dirs=$Dirs" `
      "+pilot.out=$Out"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[baselines] seed $s FAILED with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}
Write-Host "[baselines] all ten seeds complete"
