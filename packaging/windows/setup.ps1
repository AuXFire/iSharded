# Runs the same install as install.ps1, with the wheel this installer carries.
param([switch]$Service)

$env:ISHARDED_SOURCE = (Get-ChildItem -Path $PSScriptRoot -Filter "isharded-*.whl" | Select-Object -First 1).FullName
$options = @()
if ($Service) { $options += "--service" }
try {
    & (Join-Path $PSScriptRoot "install.ps1") @options
} catch {
    # This window closes with the script, and the reason should not close with it.
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "iSharded did not finish installing. Press Enter to close"
    exit 1
}
