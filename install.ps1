# Installer for windows: the four steps of docs/install.md as one command.
#
#   .\install.ps1                              # picks the torch build from the card it finds
#   .\install.ps1 -Cpu                         # force the cpu build
#   .\install.ps1 -Venv C:\venvs\transcriber   # somewhere other than .venv
#
# in order: find a python between 3.10 and 3.12 (or let uv fetch one), make a virtual
# environment, install torch 2.8.0 from the index that matches the hardware, install everything
# else, and finish by running --check, so the result is verified rather than assumed.
param(
    [switch]$Cpu,
    [string]$Venv = ".venv"
)

Set-Location -Path $PSScriptRoot

# uv brings its own python, which is the whole reason to prefer it: nothing to install by hand
# on a machine whose only python is 3.13, where diarizen's numpy 1.26.4 has no build.
$uv = (Get-Command uv -ErrorAction SilentlyContinue)

function Test-Python($exe, $extra) {
    # a probe that must not stop the script: `py -3.12` on a machine without 3.12 writes to
    # stderr, and with ErrorActionPreference = Stop that would abort the whole search.
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $code = 'import sys; print(sys.version_info[0] * 100 + sys.version_info[1])'
        $out = & $exe @($extra + @("-c", $code)) 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $n = 0
        if (-not [int]::TryParse(($out | Select-Object -Last 1), [ref]$n)) { return $false }
        return ($n -ge 310 -and $n -lt 313)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $old
    }
}

$base = $null
if (-not $uv) {
    foreach ($candidate in @(@("py", @("-3.12")), @("py", @("-3.11")), @("py", @("-3.10")),
                             @("python3.12", @()), @("python3.11", @()), @("python3.10", @()),
                             @("python", @()))) {
        $exe = $candidate[0]
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        if (Test-Python $exe $candidate[1]) {
            $base = @{ Exe = $exe; Args = $candidate[1] }
            break
        }
    }
    if ($null -eq $base) {
        Write-Host "no python between 3.10 and 3.12 found, and no uv to fetch one." -ForegroundColor Red
        Write-Host "  winget install Python.Python.3.12"
        Write-Host "  or: winget install astral-sh.uv   (then run this script again)"
        Write-Host ""
        Write-Host "why the version matters: diarizen pins numpy 1.26.4, which has no build for 3.13."
        exit 1
    }
    $version = & $base.Exe @($base.Args + @("-V"))
    Write-Host "python: $version"
} else {
    Write-Host "uv found - it will fetch python 3.12 if this machine has none"
}

# the card decides which torch build is right; pypi's windows build has no cuda at all
if ($Cpu) {
    $variant = "cpu"
    Write-Host "installing the cpu build, as asked"
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $variant = "cu128"
    Write-Host "nvidia card found - installing the cuda 12.8 build"
} else {
    $variant = "cpu"
    Write-Host "no nvidia card found - installing the cpu build (several times slower)"
}

$ErrorActionPreference = "Stop"
$index = "https://download.pytorch.org/whl/$variant"
$py = Join-Path $Venv "Scripts\python.exe"

if ($uv) {
    & uv venv --python 3.12 $Venv
    Write-Host "installing torch 2.8.0 ($variant) - the long part, a few gigabytes"
    & uv pip install --python $py torch==2.8.0 torchaudio==2.8.0 --index-url $index
    Write-Host "installing everything else"
    & uv pip install --python $py -r requirements.txt
} else {
    & $base.Exe @($base.Args + @("-m", "venv", $Venv))
    & $py -m pip install --quiet --upgrade pip
    Write-Host "installing torch 2.8.0 ($variant) - the long part, a few gigabytes"
    & $py -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url $index
    Write-Host "installing everything else"
    & $py -m pip install -r requirements.txt
}

Write-Host ""
$ErrorActionPreference = "Continue"
& $py transcribe.py --check
Write-Host ""
Write-Host "done. to use it:"
Write-Host "  $py transcribe.py `"meeting.mp4`" --language ru"
Write-Host "or let the launchers find this environment:"
Write-Host "  setx TRANSCRIBE_PYTHON `"$((Resolve-Path $py).Path)`"    # then open a new terminal"
