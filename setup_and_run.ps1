# NFTool — one-shot setup + run (Windows).
# Forces everything: installs a REAL Python (rejecting the Microsoft Store stub),
# the Visual C++ runtime PyTorch needs, a clean venv with CPU PyTorch + the ML
# stack, runs the tests, then trains every architecture on the bundled dataset.
#
#   powershell -ExecutionPolicy Bypass -File setup_and_run.ps1
#   powershell -ExecutionPolicy Bypass -File setup_and_run.ps1 --arch MLP
param([Parameter(ValueFromRemainingArguments=$true)] $DemoArgs)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Test-RealPython($exe) {
    if (-not $exe -or $exe -match '\\WindowsApps\\') { return $false }   # Store stub
    try { $v = & $exe -c "import sys;print(sys.version_info[0])" 2>$null
          return ($LASTEXITCODE -eq 0 -and $v -eq "3") } catch { return $false }
}
function Find-Py {
    $c = @()
    foreach ($n in @("py","python")) { $g = Get-Command $n -ErrorAction SilentlyContinue; if ($g) { $c += $g.Source } }
    $c += "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe","C:\Python3*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $c += $_.FullName }
    foreach ($x in $c) { if (Test-RealPython $x) { return $x } }
    return $null
}

Write-Host "==> [1/5] Ensuring a real Python is installed"
$py = Find-Py
if (-not $py) {
    winget install -e --id Python.Python.3.11 --scope user --accept-source-agreements --accept-package-agreements --silent
    $py = Find-Py
}
if (-not $py) {
    $exe = "$env:TEMP\py311.exe"
    Invoke-WebRequest "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile $exe -UseBasicParsing
    Start-Process -Wait $exe -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1"
    $py = Find-Py
}
if (-not $py) { Write-Error "Could not install Python"; exit 1 }
Write-Host ("    Python: " + $py + " (" + (& $py --version 2>&1) + ")")

Write-Host "==> [2/5] Ensuring Visual C++ runtime (required by PyTorch)"
if (-not (Test-Path "$env:SystemRoot\System32\vcruntime140_1.dll")) {
    $vc = "$env:TEMP\vc_redist.x64.exe"
    Invoke-WebRequest "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vc -UseBasicParsing
    Start-Process -FilePath $vc -ArgumentList "/quiet /norestart" -Verb RunAs -Wait
}

Write-Host "==> [3/5] Creating venv + installing CPU PyTorch + ML stack"
& $py -m venv .venv
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install --index-url https://download.pytorch.org/whl/cpu torch
& $venvPy -m pip install numpy pandas scikit-learn optuna pytest matplotlib

Write-Host "==> [4/5] Running test suite"
& $venvPy -m pytest tests -q

Write-Host "==> [5/5] Training all registered architectures on the bundled dataset"
if (-not $DemoArgs) { $DemoArgs = @("--trials","5","--epochs","40") }
& $venvPy demo.py @DemoArgs
