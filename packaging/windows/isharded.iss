; The Windows installer. Build it with:
;   iscc /DAppVersion=0.3.0 /DWheel=..\..\dist\isharded-0.3.0-py3-none-win_amd64.whl packaging\windows\isharded.iss
; It is unsigned until the project has a certificate. See "Unsigned packages" in docs/GUIDE.md.

[Setup]
AppId=iSharded
AppName=iSharded
AppVersion={#AppVersion}
AppPublisher=The iSharded project
AppPublisherURL=https://github.com/AuXFire/iSharded
; Per user, so it needs no administrator. Only the firewall task asks for one.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\iSharded\installer
DisableDirPage=yes
DefaultGroupName=iSharded
DisableProgramGroupPage=yes
OutputDir=..\..\dist
OutputBaseFilename=iSharded-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=iSharded

[Tasks]
Name: service; Description: "Start iSharded when I log in"; Flags: unchecked
Name: firewall; Description: "Let other machines on my private network reach this node through Windows Firewall. Windows asks for an administrator for this one step. Leave it off if this machine links to others over Tailscale"; Flags: unchecked

[InstallDelete]
; An upgrade would otherwise leave the old wheel here, and setup.ps1 would install it.
Type: files; Name: "{app}\isharded-*.whl"

[Files]
Source: "..\..\install.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "firewall.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "uninstall.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Wheel}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Open iSharded"; Filename: "{localappdata}\iSharded\venv\Scripts\isharded.exe"; Parameters: "up --background"
Name: "{group}\Stop iSharded"; Filename: "{localappdata}\iSharded\venv\Scripts\isharded.exe"; Parameters: "stop"
Name: "{group}\Uninstall iSharded"; Filename: "{uninstallexe}"

[Run]
; A window stays open for these, because a question about joining a mesh needs a console.
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup.ps1"""; StatusMsg: "Installing iSharded. This takes a minute or two."; Tasks: not service
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup.ps1"" -Service"; StatusMsg: "Installing iSharded. This takes a minute or two."; Tasks: service
; {localappdata} is named here because an elevated copy may run as a different account.
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\firewall.ps1"" -Python ""{localappdata}\iSharded\venv\Scripts\python.exe"""; Verb: runas; Flags: shellexec waituntilterminated; Tasks: firewall

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\uninstall.ps1"""; RunOnceId: "isharded"
