#Requires -Version 5.1
<#
.SYNOPSIS
    Molebie AI — Bootstrap Installer for Windows
.DESCRIPTION
    Ensures Python 3.10+ is available, creates a virtual environment,
    installs the molebie-ai CLI, then launches the interactive wizard.

    Usage:
      Local:   .\install.ps1                          (interactive)
               .\install.ps1 -Quick                   (auto-select defaults)
      Remote:  irm https://molebieai.com/install.ps1 | iex
               irm https://raw.githubusercontent.com/Jimmy6929/Molebie_AI/main/install.ps1 | iex
#>

param(
    [switch]$Quick,
    [string]$InstallDir = ""
)

$ErrorActionPreference = "Stop"

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
function Write-Info  { param([string]$Msg) Write-Host "[INFO] " -ForegroundColor Blue -NoNewline; Write-Host $Msg }
function Write-Ok    { param([string]$Msg) Write-Host "[OK]   " -ForegroundColor Green -NoNewline; Write-Host $Msg }
function Write-Warn  { param([string]$Msg) Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline; Write-Host $Msg }
function Write-Fail  { param([string]$Msg) Write-Host "[FAIL] " -ForegroundColor Red -NoNewline; Write-Host $Msg; exit 1 }

function Test-Command { param([string]$Name) $null -ne (Get-Command $Name -ErrorAction SilentlyContinue) }

function Get-PyMinor {
    param([string]$PythonCmd)
    try {
        $output = & $PythonCmd -c "import sys; print(sys.version_info.minor)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $output -match '^\d+$') { return [int]$output }
    } catch {}
    return 0
}

function Get-PyVersion {
    param([string]$PythonCmd)
    try {
        $output = & $PythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0) { return $output.Trim() }
    } catch {}
    return "unknown"
}

# ──────────────────────────────────────────────────────────────
# Detect execution mode: local (inside repo) vs remote
# ──────────────────────────────────────────────────────────────
$RemoteMode = $false
$ScriptDir = $PSScriptRoot

if (-not $ScriptDir -or -not (Test-Path (Join-Path $ScriptDir "pyproject.toml")) -or -not (Test-Path (Join-Path $ScriptDir "gateway"))) {
    $RemoteMode = $true
}

# ──────────────────────────────────────────────────────────────
# Remote mode: clone the repo first, then re-exec local script
# ──────────────────────────────────────────────────────────────
if ($RemoteMode) {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor White
    Write-Host "  Molebie AI - Remote Installer (Windows)" -ForegroundColor White
    Write-Host "==================================================" -ForegroundColor White
    Write-Host ""

    if (-not (Test-Command "git")) {
        Write-Fail "git is required. Install from https://git-scm.com/downloads/win"
    }

    if (-not $InstallDir) {
        $InstallDir = Join-Path $env:USERPROFILE "Molebie_AI"
    }

    if (Test-Path $InstallDir) {
        if ((Test-Path (Join-Path $InstallDir "pyproject.toml")) -and (Test-Path (Join-Path $InstallDir "gateway"))) {
            Write-Info "Existing installation found at $InstallDir"
            Write-Info "Updating with git pull..."
            git -C $InstallDir pull --ff-only 2>$null
            if ($LASTEXITCODE -eq 0) { Write-Ok "Repository updated" }
            else { Write-Warn "git pull failed - continuing with existing code" }
        } else {
            Write-Fail "$InstallDir already exists but is not a Molebie AI installation."
        }
    } else {
        Write-Info "Cloning into $InstallDir..."
        git clone --depth 1 https://github.com/Jimmy6929/Molebie_AI.git $InstallDir
        if ($LASTEXITCODE -ne 0) { Write-Fail "git clone failed" }
        Write-Ok "Repository cloned"
    }

    Write-Info "Handing off to local installer..."
    $installerPath = Join-Path $InstallDir "install.ps1"
    $reArgs = @("-ExecutionPolicy", "Bypass", "-File", $installerPath)
    if ($Quick) { $reArgs += "-Quick" }
    if ($InstallDir) { $reArgs += @("-InstallDir", $InstallDir) }
    & powershell @reArgs
    exit $LASTEXITCODE
}

# ──────────────────────────────────────────────────────────────
# Local mode: we are inside the repo
# ──────────────────────────────────────────────────────────────
Set-Location $ScriptDir

Write-Host ""
Write-Host "==================================================" -ForegroundColor White
Write-Host "  Molebie AI - Installer" -ForegroundColor White
Write-Host "==================================================" -ForegroundColor White
Write-Host ""

# ──────────────────────────────────────────────────────────────
# Step 0: Check WSL2 availability
# ──────────────────────────────────────────────────────────────
try {
    $wslOutput = & wsl --status 2>$null
    if ($LASTEXITCODE -ne 0) { throw "wsl not ready" }
} catch {
    Write-Warn "WSL2 is not installed or not enabled."
    Write-Warn "Some features require WSL2: Docker containers, local inference backends."
    Write-Info "Install WSL2:  wsl --install  (requires restart)"
    Write-Info "Continuing without WSL2 - core features will still work."
    Write-Host ""
}

# ──────────────────────────────────────────────────────────────
# Step 1: Find Python 3.10+
# ──────────────────────────────────────────────────────────────
Write-Info "Looking for Python 3.10+..."

$PythonCmd = ""

# Try Python Launcher for Windows first (py -3.X), then direct commands
$candidates = @(
    @{ cmd = "py"; args = @("-3.13", "--version"); run = "py"; runArgs = @("-3.13") },
    @{ cmd = "py"; args = @("-3.12", "--version"); run = "py"; runArgs = @("-3.12") },
    @{ cmd = "py"; args = @("-3.11", "--version"); run = "py"; runArgs = @("-3.11") },
    @{ cmd = "py"; args = @("-3.10", "--version"); run = "py"; runArgs = @("-3.10") },
    @{ cmd = "python3"; args = @("--version"); run = "python3"; runArgs = @() },
    @{ cmd = "python"; args = @("--version"); run = "python"; runArgs = @() }
)

foreach ($c in $candidates) {
    if (Test-Command $c.cmd) {
        try {
            $testArgs = $c.runArgs + @("-c", "import sys; print(sys.version_info.minor)")
            $minor = & $c.cmd @testArgs 2>$null
            if ($LASTEXITCODE -eq 0 -and $minor -match '^\d+$' -and [int]$minor -ge 10) {
                # This candidate has Python 3.10+ — store it as a callable
                $PythonCmd = $c.cmd
                $script:PythonArgs = $c.runArgs
                break
            }
        } catch {}
    }
}

if (-not $PythonCmd) {
    Write-Warn "Python 3.10+ not found."
    # Attempt auto-install (handled in Step 2b fallback below)
    # For now, try generic python
    if (Test-Command "python") { $PythonCmd = "python"; $script:PythonArgs = @() }
}

if (-not $PythonCmd) {
    Write-Fail "Python is not installed. Download from https://www.python.org/downloads/"
}

# Helper to run the selected Python
function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments)]$Args)
    & $PythonCmd @script:PythonArgs @Args
}

$PyVersion = Get-PyVersion $PythonCmd
Write-Ok "Python $PyVersion ($PythonCmd $($script:PythonArgs -join ' '))".Trim()

# ──────────────────────────────────────────────────────────────
# Step 2: Create virtual environment
# ──────────────────────────────────────────────────────────────
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

if (Test-Path $VenvPython) {
    Write-Ok "Virtual environment exists (.venv\)"
} else {
    Write-Info "Creating virtual environment..."
    Invoke-Python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create virtual environment" }
    Write-Ok "Virtual environment created (.venv\)"
}

# Upgrade pip quietly
& $VenvPython -m pip install --upgrade pip --quiet 2>$null

# ──────────────────────────────────────────────────────────────
# Step 2b: Validate Python compatibility
# ──────────────────────────────────────────────────────────────
# Probe: install the PINNED pydantic version from requirements.txt using
# only pre-built wheels.  If no wheel exists for this Python version,
# pip fails instantly.
Write-Info "Checking package compatibility..."

$reqFile = Join-Path $ScriptDir "gateway\requirements.txt"
$PydanticPin = "pydantic"
if (Test-Path $reqFile) {
    $match = Select-String -Path $reqFile -Pattern '^pydantic==' | Select-Object -First 1
    if ($match) { $PydanticPin = $match.Line.Trim() }
}

& $VenvPip install $PydanticPin --only-binary :all: --force-reinstall --quiet 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Ok "Package compatibility verified"
} else {
    Write-Warn "Python $PyVersion has no pre-built packages for key dependencies."
    Write-Warn "This usually means the Python version is too new for the package ecosystem."
    Write-Info "Looking for a compatible Python..."

    $FallbackCmd = ""
    $FallbackArgs = @()

    # Try other installed Python versions via py launcher
    foreach ($v in @("3.13", "3.12", "3.11", "3.10")) {
        if (Test-Command "py") {
            try {
                $minor = & py "-$v" -c "import sys; print(sys.version_info.minor)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $minor -match '^\d+$' -and [int]$minor -ge 10) {
                    $FallbackCmd = "py"
                    $FallbackArgs = @("-$v")
                    break
                }
            } catch {}
        }
    }

    if (-not $FallbackCmd) {
        # Attempt 1: winget (Windows 10/11)
        if (Test-Command "winget") {
            Write-Info "Installing Python 3.12 via winget..."
            winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent 2>$null
            if ($LASTEXITCODE -eq 0) {
                # Refresh PATH
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                if (Test-Command "py") {
                    try {
                        $minor = & py "-3.12" -c "import sys; print(sys.version_info.minor)" 2>$null
                        if ($LASTEXITCODE -eq 0) { $FallbackCmd = "py"; $FallbackArgs = @("-3.12") }
                    } catch {}
                }
                if (-not $FallbackCmd) {
                    $FallbackCmd = "python3.12"
                    $FallbackArgs = @()
                }
                if ($FallbackCmd) { Write-Ok "Python 3.12 installed via winget" }
            } else {
                Write-Warn "winget install failed - trying next method..."
            }
        }

        # Attempt 2: chocolatey
        if (-not $FallbackCmd -and (Test-Command "choco")) {
            Write-Info "Installing Python 3.12 via Chocolatey..."
            choco install python312 -y --no-progress 2>$null
            if ($LASTEXITCODE -eq 0) {
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                if (Test-Command "py") {
                    try {
                        $minor = & py "-3.12" -c "import sys; print(sys.version_info.minor)" 2>$null
                        if ($LASTEXITCODE -eq 0) { $FallbackCmd = "py"; $FallbackArgs = @("-3.12") }
                    } catch {}
                }
                if ($FallbackCmd) { Write-Ok "Python 3.12 installed via Chocolatey" }
            } else {
                Write-Warn "Chocolatey install failed - trying next method..."
            }
        }

        # Attempt 3: Direct download from python.org
        if (-not $FallbackCmd) {
            Write-Info "Downloading Python 3.12 from python.org..."
            $installerUrl = "https://www.python.org/ftp/python/3.12.13/python-3.12.13-amd64.exe"
            $installerPath = Join-Path $env:TEMP "molebie-python-3.12.exe"
            try {
                [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
                Write-Info "Installing Python 3.12..."
                Start-Process -FilePath $installerPath -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1" -Wait -NoNewWindow
                Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
                # Refresh PATH
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                if (Test-Command "py") {
                    try {
                        $minor = & py "-3.12" -c "import sys; print(sys.version_info.minor)" 2>$null
                        if ($LASTEXITCODE -eq 0) { $FallbackCmd = "py"; $FallbackArgs = @("-3.12") }
                    } catch {}
                }
                if ($FallbackCmd) { Write-Ok "Python 3.12 installed from python.org" }
            } catch {
                Write-Warn "Download/install failed: $_"
            }
        }

        if (-not $FallbackCmd) {
            Write-Host ""
            Write-Fail "Could not install a compatible Python automatically.`n`n  Install Python 3.12 from https://www.python.org/downloads/`n  Make sure to check 'Add Python to PATH' during installation.`n`n  Then re-run this installer."
        }
    }

    # Switch to fallback Python and recreate venv
    $PythonCmd = $FallbackCmd
    $script:PythonArgs = $FallbackArgs
    $PyVersion = Get-PyVersion $PythonCmd
    Write-Info "Switching to Python $PyVersion..."

    if (Test-Path $VenvDir) { Remove-Item $VenvDir -Recurse -Force }
    Invoke-Python -m venv $VenvDir
    & $VenvPython -m pip install --upgrade pip --quiet 2>$null

    # Re-verify
    & $VenvPip install $PydanticPin --only-binary :all: --force-reinstall --quiet 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Python $PyVersion - package compatibility verified"
    } else {
        Write-Fail "Python $PyVersion also failed the compatibility check.`n  Install Python 3.12 from https://www.python.org/downloads/ and re-run."
    }
}

# ──────────────────────────────────────────────────────────────
# Step 3: Install CLI
# ──────────────────────────────────────────────────────────────
Write-Info "Installing molebie-ai CLI..."
& $VenvPip install -e . --quiet 2>$null

$MolebieCli = Join-Path $VenvDir "Scripts\molebie-ai.exe"
if (Test-Path $MolebieCli) {
    Write-Ok "CLI installed"
} else {
    Write-Fail "CLI installation failed. Check Python and try again."
}

# ──────────────────────────────────────────────────────────────
# Step 4: Create bin\molebie-ai.cmd wrapper
# ──────────────────────────────────────────────────────────────
$binDir = Join-Path $ScriptDir "bin"
if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir | Out-Null }

$wrapperContent = @"
@echo off
"%~dp0..\.venv\Scripts\molebie-ai.exe" %*
"@
Set-Content -Path (Join-Path $binDir "molebie-ai.cmd") -Value $wrapperContent
Write-Ok "Created bin\molebie-ai.cmd wrapper"

# ──────────────────────────────────────────────────────────────
# Step 5: Launch the interactive wizard
# ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Info "Launching setup wizard..."
Write-Host ""

if ($Quick) {
    & $MolebieCli install --quick
} else {
    & $MolebieCli install
}
