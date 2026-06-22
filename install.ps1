<#
  NFTool Windows installer
  Usage:  irm https://raw.githubusercontent.com/nm-z/NFTool/windows/install.ps1 | iex

  Downloads the latest Windows build, installs it (bundling the Microsoft Edge
  WebView2 runtime), creates a desktop shortcut, and launches the app.

  Set $env:NFTOOL_SRC to a base URL to install from a mirror / local file server
  instead of GitHub (expects <NFTOOL_SRC>/NFTool-setup.exe).
#>
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = 'nm-z/NFTool'

if ($env:NFTOOL_SRC) {
    $url  = "$($env:NFTOOL_SRC.TrimEnd('/'))/NFTool-setup.exe"
    $name = 'NFTool-setup.exe'
} else {
    Write-Host 'Finding latest NFTool release...' -ForegroundColor Cyan
    $headers = @{ 'User-Agent' = 'NFTool-Installer'; 'Accept' = 'application/vnd.github+json' }
    $asset = $null
    foreach ($r in (Invoke-RestMethod "https://api.github.com/repos/$repo/releases?per_page=30" -Headers $headers)) {
        $asset = $r.assets | Where-Object { $_.name -like '*x64-setup.exe' } | Select-Object -First 1
        if ($asset) { Write-Host "  found $($asset.name) in $($r.tag_name)" -ForegroundColor DarkGray; break }
    }
    if (-not $asset) { throw 'No x64-setup.exe asset found in the latest releases.' }
    $url  = $asset.browser_download_url
    $name = $asset.name
}

$tmp = Join-Path $env:TEMP $name
Write-Host "Downloading $name ..." -ForegroundColor Cyan
Invoke-WebRequest $url -OutFile $tmp -UseBasicParsing

Write-Host 'Installing NFTool (this also installs the WebView2 runtime)...' -ForegroundColor Cyan
Start-Process $tmp -ArgumentList '/S' -Wait

$exe = Join-Path $env:LOCALAPPDATA 'NFTool\NFTool.exe'
if (-not (Test-Path $exe)) {
    $hit = Get-ChildItem "$env:LOCALAPPDATA\NFTool" -Filter 'NFTool.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { $exe = $hit.FullName }
}
if (-not (Test-Path $exe)) { throw 'Install finished but NFTool.exe was not found.' }

$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'NFTool.lnk'
$sh  = New-Object -ComObject WScript.Shell
$s   = $sh.CreateShortcut($lnk)
$s.TargetPath       = $exe
$s.WorkingDirectory = Split-Path $exe
$s.IconLocation     = $exe
$s.Save()

Write-Host 'NFTool installed. Desktop shortcut created. Launching...' -ForegroundColor Green
Start-Process $exe
