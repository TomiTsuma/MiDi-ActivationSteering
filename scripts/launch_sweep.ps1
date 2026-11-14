# scripts/launch_sweep.ps1 -- the main dose-response sweep with the Section
# 3.8 controls. 23 cells, 500 molecules each, 11,500 total, about 14.8 h at
# the measured steered rate of 778 mol/h.
#
# The design and the reason for every coefficient are in
# results/sweep_plan.md. Resumable: completed arms are skipped and chunks are
# written atomically, so re-running this after a crash costs at most 50
# molecules and never duplicates an arm.
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Ckpt = Join-Path $Repo "midi\checkpoints\checkpoint_qm9_h_adaptive.ckpt"
$Out = Join-Path $Repo "results\sweep"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Set-Location (Join-Path $Repo "midi")
python3.11 -u ..\scripts\run_pilot.py `
  dataset=qm9 dataset.remove_h=False +experiment=qm9_with_h_adaptive `
  general.wandb=disabled general.gpus=1 `
  "general.test_only=$Ckpt" `
  "+pilot.arms=[dom,rand,shuf]" `
  "+pilot.coeffs=[-160,-80,-40,-20,-10,10,20,40,80,160]" `
  "+pilot.control_coeffs=[-80,-40,-20,20,40,80]" `
  +pilot.n=500 +pilot.window=late `
  "+pilot.store=$Repo\activations_test_full" `
  "+pilot.dirs=$Repo\results\directions" `
  "+pilot.out=$Out"
