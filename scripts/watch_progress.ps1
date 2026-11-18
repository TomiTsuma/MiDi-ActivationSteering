<#
.SYNOPSIS
    Read-only progress view for a running Phase One stage.
.DESCRIPTION
    Shows, for the newest run: the MiDi process, GPU state, the active
    orchestration log's tail, molecules started (counted from "Sampling a batch
    with N graphs" lines), and the newest files in the Hydra run directory.

    Note: molecule counting only works if the run was launched by a script that
    passes -u to python. Without it, Python block-buffers stdout through the
    pipe and the log stays empty until the process exits.
.EXAMPLE
    .\scripts\watch_progress.ps1
.EXAMPLE
    .\scripts\watch_progress.ps1 -Loop -Every 120
#>
[CmdletBinding()]
param(
    [switch] $Loop,
    [int]    $Every = 60,
    [int]    $Tail  = 8
)

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Show-Once {
    Write-Host ''
    Write-Host ("==== " + (Get-Date -Format 'HH:mm:ss') + " ====") -ForegroundColor Cyan

    # --- the MiDi process -------------------------------------------------
    $procs = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
               Where-Object { $_.CommandLine -and
                             ($_.CommandLine -like '*main.py*' -or
                              $_.CommandLine -like '*capture.py*' -or
                              $_.CommandLine -like '*test_pairing.py*') })
    if ($procs.Count -eq 0) {
        Write-Host '  process : none running (finished, or not started)'
    } else {
        foreach ($p in $procs) {
            $ps = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
            $el = (Get-Date) - $p.CreationDate
            $script = 'unknown'
            foreach ($n in @('main.py','capture.py','test_pairing.py')) {
                if ($p.CommandLine -like "*$n*") { $script = $n }
            }
            Write-Host ("  process : pid {0}  {1}  elapsed {2:hh\:mm\:ss}  cpu {3:N0}s  rss {4:N0} MB" -f `
                        $p.ProcessId, $script, $el, $ps.CPU, ($ps.WS / 1MB))
        }
    }

    # --- GPU --------------------------------------------------------------
    $g = (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null)
    if ($g) { Write-Host ("  gpu     : " + ($g -join ' ')) }

    # --- newest orchestration log ----------------------------------------
    $ld = Get-ChildItem (Join-Path $Repo 'logs') -Directory -ErrorAction SilentlyContinue |
          Sort-Object Name -Descending | Select-Object -First 1
    if (-not $ld) { Write-Host '  logs    : none'; return }
    $log = Get-ChildItem $ld.FullName -Filter '*.log' -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($log) {
        $age = ((Get-Date) - $log.LastWriteTime)
        Write-Host ("  log     : {0}\{1}  ({2:N0} KB, last write {3:N0}s ago)" -f `
                    $ld.Name, $log.Name, ($log.Length / 1KB), $age.TotalSeconds)
        $txt = Get-Content $log.FullName -ErrorAction SilentlyContinue
        $hits = @($txt | Select-String -Pattern 'Sampling a batch with (\d+) graphs')
        if ($hits.Count -gt 0) {
            $done = 0
            foreach ($h in $hits) { $done += [int]$h.Matches[0].Groups[1].Value }
            Write-Host ("  batches : {0}   molecules started: {1:N0}" -f $hits.Count, $done)
        } else {
            Write-Host '  batches : no "Sampling a batch" lines yet'
            Write-Host '            (if the process is busy, the log is buffered: relaunch with -u)'
        }
        Write-Host "  --- last $Tail non-empty log lines ---"
        $txt | Where-Object { $_.Trim() } | Select-Object -Last $Tail |
            ForEach-Object { Write-Host ('    ' + $_.Substring(0, [Math]::Min(150, $_.Length))) }
    }

    # --- newest Hydra run dir --------------------------------------------
    $rd = Get-ChildItem (Join-Path $Repo 'outputs') -Directory -ErrorAction SilentlyContinue |
          Sort-Object Name -Descending | Select-Object -First 1
    if ($rd) {
        $sub = Get-ChildItem $rd.FullName -Directory -ErrorAction SilentlyContinue |
               Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($sub) {
            $files = @(Get-ChildItem $sub.FullName -Recurse -File -ErrorAction SilentlyContinue)
            $newest = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 3
            Write-Host ("  rundir  : outputs\{0}\{1}  ({2} files)" -f $rd.Name, $sub.Name, $files.Count)
            foreach ($f in $newest) {
                Write-Host ("            {0:HH:mm:ss}  {1,10:N0} B  {2}" -f `
                            $f.LastWriteTime, $f.Length, $f.Name)
            }
        }
    }
}

if ($Loop) {
    Write-Host "watching every $Every s; Ctrl-C to stop"
    while ($true) { Show-Once; Start-Sleep -Seconds $Every }
} else {
    Show-Once
}
