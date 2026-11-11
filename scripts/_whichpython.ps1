# Find which Python interpreter has the full stack and the midi package.
# Read-only. Runs _probe_interp.py rather than python -c, because PowerShell 5.1
# mangles double quotes inside native-command arguments.
$ErrorActionPreference = 'Continue'
$repo  = Split-Path -Parent $PSScriptRoot
$probe = Join-Path $PSScriptRoot '_probe_interp.py'

$candidates = @(
    'python3.11', 'python3', 'python', 'py',
    'C:\Program Files\Python38\python.exe',
    'C:\Python314\python.exe',
    "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.11.exe"
)

Write-Output ('repo  : ' + $repo)
Write-Output ('probe : ' + $probe + '   exists: ' + (Test-Path $probe))
Write-Output ''

$seen = @{}
foreach ($c in $candidates) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if (-not $cmd) { Write-Output ("{0,-58} not found" -f $c); continue }

    Push-Location $repo
    $raw = (& $c $probe 2>&1 | Out-String)
    $code = $LASTEXITCODE
    Pop-Location

    $lines = @($raw -split "`r?`n" | Where-Object { $_.Trim() })
    $ver = @($lines | Where-Object { $_ -like 'VER|*' })
    if ($ver.Count -eq 0) {
        Write-Output ("{0,-58} unusable (exit {1})" -f $c, $code)
        if ($lines.Count) { Write-Output ('    ' + $lines[-1].Trim()) }
        continue
    }
    $vp  = $ver[0].Split('|')
    $exe = $vp[2]
    if ($seen.ContainsKey($exe)) {
        Write-Output ("{0,-58} -> same interpreter as {1}" -f $c, $seen[$exe]); continue
    }
    $seen[$exe] = $c

    Write-Output ("{0,-58} python {1}" -f $c, $vp[1])
    Write-Output ('    exe  ' + $exe)
    foreach ($tag in @('DEPS', 'MIDI')) {
        $row = @($lines | Where-Object { $_ -like "$tag|*" })
        if ($row.Count) { Write-Output ('    ' + $row[0].Replace('|', '  ')) }
        else            { Write-Output ("    $tag  no output") }
    }
    Write-Output ''
}
Write-Output 'Pass the winning interpreter to run_phase1.ps1 with:  -Py <exe path>'
