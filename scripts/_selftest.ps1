# Self-test for run_phase1.ps1. Read-only: parses the script with PowerShell's
# own parser, checks comment-based help, and runs -List. Creates nothing.
$ErrorActionPreference = 'Continue'
$target = Join-Path $PSScriptRoot 'run_phase1.ps1'

Write-Output ('PSVersion   : ' + $PSVersionTable.PSVersion.ToString())
Write-Output ('PSEdition   : ' + $PSVersionTable.PSEdition)
Write-Output ('target      : ' + $target)
Write-Output ('exists      : ' + (Test-Path $target))
Write-Output ''

# 1. parse with the real parser
$tokens = $null; $errors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile($target, [ref]$tokens, [ref]$errors)
Write-Output ('[1] parse errors : ' + $errors.Count)
if ($errors.Count -gt 0) {
    foreach ($e in $errors) {
        Write-Output ('    line ' + $e.Extent.StartLineNumber + ' col ' + $e.Extent.StartColumnNumber + ': ' + $e.Message)
    }
} else {
    Write-Output '    PARSE CLEAN'
}
Write-Output ('    tokens parsed  : ' + $tokens.Count)
Write-Output ''

# 2. comment-based help must be discoverable
try {
    $h = Get-Help $target -ErrorAction Stop
    $syn = ($h.Synopsis -replace '\s+', ' ').Trim()
    if ($syn.Length -gt 60) { $syn = $syn.Substring(0, 60) + '...' }
    Write-Output ('[2] help synopsis: ' + $syn)
    Write-Output ('    examples      : ' + @($h.Examples.Example).Count)
} catch {
    Write-Output ('[2] Get-Help FAILED: ' + $_.Exception.Message)
}
Write-Output ''

# 3. the declared parameters
try {
    $cmd = Get-Command $target -ErrorAction Stop
    $names = @($cmd.Parameters.Keys | Where-Object {
        $_ -notin @('Verbose','Debug','ErrorAction','WarningAction','InformationAction',
                    'ErrorVariable','WarningVariable','InformationVariable','OutVariable',
                    'OutBuffer','PipelineVariable','ProgressAction')
    })
    Write-Output ('[3] parameters   : ' + ($names -join ', '))
} catch {
    Write-Output ('[3] Get-Command FAILED: ' + $_.Exception.Message)
}
Write-Output ''

# 4. -List must print exactly the twelve stages and create nothing
$logsBefore = @(Get-ChildItem -Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'logs') -Directory -ErrorAction SilentlyContinue).Count
$out = & $target -List 2>&1
$stages = @($out | Where-Object { $_ -match '^\d\d_' })
$logsAfter = @(Get-ChildItem -Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'logs') -Directory -ErrorAction SilentlyContinue).Count
Write-Output ('[4] -List stages : ' + $stages.Count + ' (expected 12)')
Write-Output ('    ' + ($stages -join ' '))
$leak = '  LEAKED A DIR'
if ($logsAfter -eq $logsBefore) { $leak = '  (created nothing, good)' }
Write-Output ('    log dirs before/after: ' + $logsBefore + '/' + $logsAfter + $leak)
Write-Output ''

# 5. an unknown stage name must be rejected
& $target -Only 99_nope *> $null
$rejected = ($LASTEXITCODE -eq 1)
Write-Output ('[5] rejects unknown stage: ' + $rejected + '  (exit code ' + $LASTEXITCODE + ')')
Write-Output '    note: Die uses Write-Host, which PS 5.1 does not route to the'
Write-Output '    output stream, so the exit code is what is checked here.'
Write-Output ''
Write-Output 'SELFTEST DONE'
