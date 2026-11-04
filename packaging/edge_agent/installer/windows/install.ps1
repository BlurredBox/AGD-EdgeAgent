<#
 BlurredBox edge agent: one-command installer for Windows Server. NOT TESTED on real Windows (Arameh owns Windows runs).

   powershell -ExecutionPolicy Bypass -File .\installer\windows\install.ps1 [-NoSetup] [-Uninstall] [-Python C:\Python312\python.exe] `
       [-- setup flags: --console-url ... --tenant-id ... (same as `blurredbox-edge setup`)]

 Needs an elevated prompt and Python 3.12 (python.org installer). Offline: pip runs with --no-index against the kit's
 wheelhouse\ (which must hold Windows wheels; build it on Windows with the same Python: scripts\build_edge_wheelhouse.sh
 is Linux-only, so use `pip download --only-binary=:all: -r requirements-edge.txt -d wheelhouse` there, plus pywin32).
 Layout: %ProgramData%\BlurredBoxEdge\{app,conf,data}. The service runs as LocalService; conf\ and data\ are ACL'd to
 SYSTEM, Administrators and LocalService. Uninstall removes the service and app\, and leaves conf\ and data\.
 Secrets: AGD_HMAC_SECRET and BLURREDBOX_API_KEY from the environment, or the hidden prompt of `blurredbox-edge setup`.
#>
param([switch]$NoSetup, [switch]$Uninstall, [string]$Python = "", [string]$Kit = "", [string]$Wheelhouse = "",
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$SetupArgs)
$ErrorActionPreference = "Stop"
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole("Administrators")) { throw "run from an elevated prompt" }
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Kit) { $Kit = Split-Path -Parent (Split-Path -Parent $Here) }       # kit\edge
$Root = Join-Path $env:ProgramData "BlurredBoxEdge"; $App = "$Root\app"; $Conf = "$Root\conf"; $Data = "$Root\data"
$Svc = "blurredbox-edge"

if ($Uninstall) {
    if (Get-Service $Svc -ErrorAction SilentlyContinue) { Stop-Service $Svc -ErrorAction SilentlyContinue; & "$App\venv\Scripts\python.exe" "$App\lib\blurredbox_edge_service.py" remove }
    Remove-Item -Recurse -Force $App -ErrorAction SilentlyContinue
    Write-Host "Removed the program and the service. Left untouched: $Conf and $Data."
    exit 0
}
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
& $Python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"; if ($LASTEXITCODE) { throw "Python 3.12 is required ($Python)" }
$Wheel = (Get-ChildItem "$Kit\blurredbox_edge_agent-*.whl" | Select-Object -First 1).FullName
if (-not $Wheel) { throw "edge wheel not found in $Kit" }
if (-not $Wheelhouse) { $Wheelhouse = "$Kit\wheelhouse" }
if (-not (Test-Path $Wheelhouse)) { throw "no wheelhouse at $Wheelhouse (offline install needs it)" }

New-Item -ItemType Directory -Force $App, "$App\lib", $Conf, $Data, "$Data\state" | Out-Null
$was = Get-Service $Svc -ErrorAction SilentlyContinue
if ($was -and $was.Status -eq "Running") { Stop-Service $Svc }
if (Test-Path "$App\venv") { Remove-Item -Recurse -Force "$App\venv" }
& $Python -m venv "$App\venv"
& "$App\venv\Scripts\python.exe" -m pip install --no-index --find-links $Wheelhouse --quiet -r "$Kit\requirements-edge.txt" blurredbox-agd pywin32 $Wheel
if ($LASTEXITCODE) { throw "pip install failed" }
Copy-Item "$Kit\partner_pair.py" "$App\lib\partner_pair.py" -Force
Copy-Item "$Here\blurredbox_edge_service.py" "$App\lib\blurredbox_edge_service.py" -Force
Copy-Item "$Here\..\blurredbox-edge" "$App\lib\blurredbox_edge_cli.py" -Force
Set-Content "$App\blurredbox-edge.cmd" "@echo off`r`n`"$App\venv\Scripts\python.exe`" `"$App\lib\blurredbox_edge_cli.py`" %*"
if (-not (Test-Path "$Conf\config.json")) {
    '{"tenant_id":"<TENANT_ID>","install_id":"<INSTALL_ID>","models":{"<YOUR_MODEL_NAME>":"<BLURREDBOX_MODEL_ID>"},"secret_env":"AGD_HMAC_SECRET","endpoints":["<CONSOLE_URL>"],"state_dir":"' + ($Data -replace '\\','\\') + '\\state","cadence_seconds":300,"output_kind":"<probability|signed>","source":{"type":"jsonl","path":"<PATH_TO_MODEL_LOG.jsonl>"}}' | Set-Content "$Conf\config.json"
}
foreach ($d in $Conf, $Data) { icacls $d /inheritance:r /grant "SYSTEM:(OI)(CI)F" "Administrators:(OI)(CI)F" "LOCAL SERVICE:(OI)(CI)M" | Out-Null }
& "$App\venv\Scripts\python.exe" "$App\lib\blurredbox_edge_service.py" --startup auto install
sc.exe config $Svc obj= "NT AUTHORITY\LocalService" | Out-Null
sc.exe failure $Svc reset= 86400 actions= restart/10000/restart/10000/restart/10000 | Out-Null
if ($was -and $was.Status -eq "Running") { Start-Service $Svc }
Write-Host "Installed. Next: & `"$App\blurredbox-edge.cmd`" setup"
if (-not $NoSetup) { & "$App\blurredbox-edge.cmd" setup @SetupArgs; exit $LASTEXITCODE }
