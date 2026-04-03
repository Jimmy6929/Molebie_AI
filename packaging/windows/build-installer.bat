@echo off
REM ══════════════════════════════════════════════════════════════
REM Build molebie-ai Windows installer using NSIS
REM ══════════════════════════════════════════════════════════════
REM Requires: NSIS 3.x installed (https://nsis.sourceforge.io)
REM           PyInstaller binary at dist\molebie-ai.exe
REM Output:   dist\molebie-ai-0.1.0-setup.exe
REM
REM Usage: packaging\windows\build-installer.bat

setlocal

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..

if not exist "%PROJECT_ROOT%\dist\molebie-ai.exe" (
    echo ERROR: dist\molebie-ai.exe not found.
    echo Run 'bash packaging/pyinstaller/build.sh' first.
    exit /b 1
)

where makensis >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: makensis not found. Install NSIS from https://nsis.sourceforge.io
    exit /b 1
)

echo === Building Windows installer ===
makensis "%SCRIPT_DIR%molebie-ai.nsi"

if %ERRORLEVEL% equ 0 (
    echo.
    echo === Installer built ===
    echo Output: dist\molebie-ai-0.1.0-setup.exe
) else (
    echo ERROR: NSIS build failed
    exit /b 1
)
