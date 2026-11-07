# Adds or removes the Windows Firewall rules for a node. It needs an administrator.
#
# The first rule opens a node's ports and those of its lanes, and UDP 8208 for the search for
# a mesh. Direct links between stages take any free port, so the second rule allows the
# ports Windows hands out, for this install's Python only. Both apply on private and
# domain networks, never on public ones. A node on another --port needs its own rule.
param(
    [string]$Python,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$Name = "iSharded"

if ($Remove) {
    netsh advfirewall firewall delete rule name="$Name" | Out-Null
    netsh advfirewall firewall delete rule name="$Name direct links" | Out-Null
    return
}

# A venv's python.exe only starts the real one, and the real one is what listens.
$real = & $Python -c "import sys; print(sys._base_executable)"
if ($LASTEXITCODE -ne 0 -or -not $real) { throw "$Python did not run, so no rule was added." }

$tcp = (0..7 | ForEach-Object { "$(8200 + 1000 * $_)-$(8209 + 1000 * $_)" }) -join ","
netsh advfirewall firewall delete rule name="$Name" | Out-Null
netsh advfirewall firewall add rule name="$Name" dir=in action=allow protocol=TCP localport=$tcp profile=private,domain
if ($LASTEXITCODE -ne 0) { throw "Windows refused the firewall rule." }
netsh advfirewall firewall add rule name="$Name" dir=in action=allow protocol=UDP localport=8208 profile=private,domain
if ($LASTEXITCODE -ne 0) { throw "Windows refused the firewall rule." }
netsh advfirewall firewall delete rule name="$Name direct links" | Out-Null
netsh advfirewall firewall add rule name="$Name direct links" dir=in action=allow protocol=TCP localport=49152-65535 program="$real" profile=private,domain
if ($LASTEXITCODE -ne 0) { throw "Windows refused the firewall rule." }
