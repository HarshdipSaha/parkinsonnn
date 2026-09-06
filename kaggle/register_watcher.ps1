# Registers a Task Scheduler job that runs scripts/watch_kaggle_output.py every 5 minutes (survives closed Claude sessions).
#   powershell -File kaggle/register_watcher.ps1          # register (idempotent)
#   schtasks /Query /TN DaT-KaggleOutputWatch /V /FO LIST # inspect
#   schtasks /Delete /TN DaT-KaggleOutputWatch /F         # remove
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path (Split-Path -Parent (Get-Command python).Source) "pythonw.exe"
if (-not (Test-Path $py)) { $py = (Get-Command python).Source }
$script = Join-Path $root "scripts\watch_kaggle_output.py"
$tr = "`"$py`" `"$script`""
schtasks /Create /SC MINUTE /MO 5 /TN "DaT-KaggleOutputWatch" /TR $tr /F | Out-Host
schtasks /Run /TN "DaT-KaggleOutputWatch" | Out-Host
