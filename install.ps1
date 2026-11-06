# Install ShardMesh for this user on Windows. Nothing here needs an administrator.
$ErrorActionPreference = "Stop"

$folder = if ($env:SHARDMESH_INSTALL_DIR) { $env:SHARDMESH_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "ShardMesh\venv" }
$source = if ($env:SHARDMESH_SOURCE) { $env:SHARDMESH_SOURCE } else { "https://github.com/AuXFire/shardmesh/archive/refs/heads/main.tar.gz" }

$python = (Get-Command py -ErrorAction SilentlyContinue).Source
$arguments = @("-3")
if (-not $python) {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    $arguments = @()
}
if (-not $python) { throw "ShardMesh needs Python 3.10 or newer. Install it from python.org, then run this again." }
& $python @arguments -c "import sys; sys.exit(sys.version_info < (3, 10))"
if ($LASTEXITCODE -ne 0) { throw "ShardMesh needs Python 3.10 or newer. Install it from python.org, then run this again." }

& $python @arguments -m venv $folder
$venv = Join-Path $folder "Scripts\python.exe"
# Antivirus that inspects TLS breaks pip's own certificate checks. Windows' store knows its certificate.
$pip = @("-m", "pip", "install", "--quiet", "--use-feature=truststore")
& $venv @pip --upgrade pip

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & $venv @pip torch --index-url https://download.pytorch.org/whl/cu124
} else {
    & $venv @pip torch --index-url https://download.pytorch.org/whl/cpu
}
& $venv @pip $source
if ($LASTEXITCODE -ne 0) { throw "pip could not install ShardMesh. See the message above." }

$scripts = Join-Path $folder "Scripts"
$path = [Environment]::GetEnvironmentVariable("Path", "User")
if (($path -split ";") -notcontains $scripts) {
    [Environment]::SetEnvironmentVariable("Path", "$path;$scripts", "User")
    Write-Host "Added ShardMesh to your PATH. Open a new terminal for that to take effect."
}
Write-Host "Installed. Start with: shardmesh up"
