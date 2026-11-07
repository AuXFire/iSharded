# Install iSharded for this user on Windows, and start it. Nothing here needs an administrator or Python.
#
#   irm https://raw.githubusercontent.com/AuXFire/iSharded/main/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/AuXFire/iSharded/main/install.ps1))) --no-start
#   ... --service     also start at login

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # the progress bar makes large downloads many times slower

$UvVersion = "0.13.6"
$UvAsset = "uv-x86_64-pc-windows-msvc.zip"  # Windows on Arm runs this one too
$UvHash = "a2a8820d5eea32f08ef467ef3b7f66883a9b076d5737a2fc2fbb5f458d1bb50f"
$Repository = "https://github.com/AuXFire/iSharded"
$PythonVersion = "3.12"
$WheelTag = "win_amd64"

# "hash name" lines from a SHA256SUMS file. Answers this machine's wheel, the pure wheel, or nothing.
function Select-Wheel([string[]]$Lines, [string]$Tag) {
    $pure = $null
    foreach ($line in $Lines) {
        $hash, $name = -split $line
        if (-not $name) { continue }
        $name = $name.TrimStart("*")
        if ($name.EndsWith("-py3-none-$Tag.whl")) { return @{ Hash = $hash; Name = $name } }
        if ($name -match "^isharded-.*-py3-none-any\.whl$") { $pure = @{ Hash = $hash; Name = $name } }
    }
    return $pure
}

function Assert-Hash([string]$Path, [string]$Expected) {
    $found = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($found -ne $Expected.ToLowerInvariant()) {
        throw "$(Split-Path -Leaf $Path) does not match its SHA-256, so it was not installed. Expected $Expected, got $found."
    }
}

function Invoke-Native([string]$Program) {
    & $Program @args
    if ($LASTEXITCODE -ne 0) { throw "$(Split-Path -Leaf $Program) $($args -join ' ') failed. See the message above." }
}

function Get-Uv([string]$Root, [string]$Work) {
    $folder = Join-Path $Root "uv\$UvVersion"
    $uv = Join-Path $folder "uv.exe"
    if (Test-Path -LiteralPath $uv) { return $uv }
    Write-Host "Fetching uv $UvVersion, which provides Python."
    $base = if ($env:ISHARDED_UV_URL) { $env:ISHARDED_UV_URL } else { "https://github.com/astral-sh/uv/releases/download/$UvVersion" }
    $archive = Join-Path $Work $UvAsset
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$UvAsset" -OutFile $archive
    Assert-Hash $archive $UvHash
    Expand-Archive -LiteralPath $archive -DestinationPath $folder -Force
    return $uv
}

function Write-FallbackNote([string]$Reason) {
    Write-Host "$Reason So this installs from the source archive, without the native engine: the fast path is off."
    Write-Host "To get it, run this again once a release has a wheel for this machine, or build it from a checkout with: python scripts\build_engine.py"
}

function Get-Package([string]$Work) {
    if ($env:ISHARDED_SOURCE) { return $env:ISHARDED_SOURCE }
    $archive = "$Repository/archive/refs/heads/main.tar.gz"
    $releases = if ($env:ISHARDED_RELEASE_URL) { $env:ISHARDED_RELEASE_URL } else { "$Repository/releases/latest/download" }
    $sums = Join-Path $Work "SHA256SUMS"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$releases/SHA256SUMS" -OutFile $sums
    } catch {
        Write-FallbackNote "No release is reachable."
        return $archive
    }
    $wheel = Select-Wheel (Get-Content -LiteralPath $sums) $WheelTag
    if (-not $wheel) {
        Write-FallbackNote "The latest release has no wheel for this machine."
        return $archive
    }
    Write-Host "Fetching $($wheel.Name)."
    $file = Join-Path $Work $wheel.Name
    Invoke-WebRequest -UseBasicParsing -Uri "$releases/$($wheel.Name)" -OutFile $file
    Assert-Hash $file $wheel.Hash
    return $file
}

function Add-ToPath([string]$Scripts) {
    $path = [Environment]::GetEnvironmentVariable("Path", "User")
    if ((("$path") -split ";") -notcontains $Scripts) {
        [Environment]::SetEnvironmentVariable("Path", ((@($path, $Scripts) | Where-Object { $_ }) -join ";"), "User")
        Write-Host "Added iSharded to your PATH. Open a new terminal for that to take effect."
    }
    $env:Path = "$env:Path;$Scripts"
}

function Install-ISharded([string[]]$Options) {
    $start = $true
    $service = $false
    foreach ($option in $Options) {
        switch ($option) {
            "--no-start" { $start = $false }
            "--service" { $service = $true }
            default { throw "Unknown option $option. This takes --no-start or --service." }
        }
    }
    if ($service -and -not $start) { throw "--no-start and --service do not go together: a service starts the node." }

    # Older Windows PowerShell offers TLS 1.0 first, which GitHub refuses.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $folder = if ($env:ISHARDED_INSTALL_DIR) { $env:ISHARDED_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "iSharded\venv" }
    $root = Split-Path -Parent $folder
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    $work = Join-Path ([IO.Path]::GetTempPath()) ("isharded-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $work | Out-Null
    try {
        # uv's Python stays beside the install, so removing that folder removes all of it.
        if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $root "python" }
        # Antivirus that inspects TLS breaks uv's bundled certificates. For pip the same fix
        # is --use-feature=truststore. Windows' store knows the antivirus's certificate.
        if (-not $env:UV_SYSTEM_CERTS) { $env:UV_SYSTEM_CERTS = "1" }

        # A node still running would keep the old code, and Windows will not replace a file in use.
        $old = Join-Path $folder "Scripts\isharded.exe"
        if (Test-Path -LiteralPath $old) { & $old stop }
        $uv = Get-Uv $root $work
        $python = Join-Path $folder "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $python)) {
            # --seed keeps pip in the environment, for anyone adding packages by hand.
            Invoke-Native $uv venv --quiet --seed --clear --managed-python --python $PythonVersion $folder
        }
        $index = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cu124" } else { "cpu" }
        Invoke-Native $uv pip install --python $python torch --default-index "https://download.pytorch.org/whl/$index"
        $package = Get-Package $work
        Invoke-Native $uv pip install --python $python --reinstall-package isharded --refresh-package isharded $package
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }

    $scripts = Join-Path $folder "Scripts"
    Add-ToPath $scripts
    & $python -I -c "import sys; from isharded.engine import find_library; sys.exit(find_library() is None)"
    if ($LASTEXITCODE -ne 0) { Write-Host "This install has no native engine, so the fast path is off. isharded doctor says how to get it." }

    if (-not $start) {
        Write-Host "Installed. Start with: isharded up"
        return
    }
    $isharded = Join-Path $scripts "isharded.exe"
    if ($service) {
        & $isharded service install
        if ($LASTEXITCODE -ne 0) { Write-Host "Could not set up the start at login. The node still runs for now." }
    }
    Invoke-Native $isharded up --background
}

# Tests load the functions above without installing anything.
if ($env:ISHARDED_INSTALL_TEST -ne "1") { Install-ISharded $args }
