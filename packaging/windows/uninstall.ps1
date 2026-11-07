# Removes what the installer added. The mesh's folder, iSharded\up, stays: it holds the mesh's keys.
$root = Join-Path $env:LOCALAPPDATA "iSharded"
$scripts = Join-Path $root "venv\Scripts"
$isharded = Join-Path $scripts "isharded.exe"

if (Test-Path -LiteralPath $isharded) {
    & $isharded stop
    & $isharded service remove 2>$null
}
$path = [Environment]::GetEnvironmentVariable("Path", "User")
if ($path) {
    [Environment]::SetEnvironmentVariable("Path", ((($path -split ";") | Where-Object { $_ -and $_ -ne $scripts }) -join ";"), "User")
}
netsh advfirewall firewall show rule name="iSharded" *> $null
if ($LASTEXITCODE -eq 0) {
    # Only here does removing need an administrator.
    Start-Process powershell.exe -Verb RunAs -Wait -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSScriptRoot\firewall.ps1`"", "-Remove"
}
foreach ($name in "venv", "python", "uv") {
    Remove-Item -LiteralPath (Join-Path $root $name) -Recurse -Force -ErrorAction SilentlyContinue
}
