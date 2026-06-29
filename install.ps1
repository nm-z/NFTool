# NFTool (claude-rewrite) — standalone one-command installer for Windows.
#   irm https://raw.githubusercontent.com/nm-z/NFTool/claude-rewrite/install.ps1 | iex
# Downloads this branch, then runs setup_and_run.ps1 (installs a real Python +
# VC++ runtime + CPU PyTorch into a venv, runs the tests, trains the model).
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = Join-Path $env:TEMP 'nftool-claude-rewrite.zip'
$dst = Join-Path $env:USERPROFILE 'NFTool-claude-rewrite'
Write-Host '==> Downloading NFTool (claude-rewrite) from GitHub...' -ForegroundColor Cyan
Invoke-WebRequest 'https://github.com/nm-z/NFTool/archive/refs/heads/claude-rewrite.zip' -OutFile $zip
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
Expand-Archive $zip $dst -Force
$root = (Get-ChildItem $dst -Directory | Select-Object -First 1).FullName
Set-Location $root
Write-Host "==> Running setup_and_run.ps1 in $root" -ForegroundColor Cyan
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_and_run.ps1 --arch MLP
