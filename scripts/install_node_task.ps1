# Run an iSharded node at startup, restarting it if it stops. Run this once, as
# administrator, on the machine that should host the node.
#
# Example:
#   .\install_node_task.ps1 -Python C:\venv\Scripts\python.exe -AppFolder C:\isharded `
#       -Arguments "-m isharded node --node-id node-b --join http://10.0.0.5:8209"
#
# To remove it:  Unregister-ScheduledTask -TaskName iShardedNode -Confirm:$false
#
# The task runs whether or not anyone is signed in, and asks for this account's
# password so that it can. A node's services restart themselves, so this only
# has to get the node going.

#Requires -RunAsAdministrator
param(
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$AppFolder,
    [Parameter(Mandatory)][string]$Arguments,
    [string]$TaskName = "iShardedNode"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Python)) { throw "No Python at $Python" }
if (-not (Test-Path $AppFolder)) { throw "No folder $AppFolder" }

$action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $AppFolder
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$credential = Get-Credential -UserName "$env:USERDOMAIN\$env:USERNAME" -Message "This account will run the node"
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -User $credential.UserName -Password $credential.GetNetworkCredential().Password -RunLevel Limited -Force | Out-Null
Write-Host "Installed $TaskName. It starts at the next boot, or now with: Start-ScheduledTask -TaskName $TaskName"
