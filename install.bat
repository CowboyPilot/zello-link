@echo off
rem =====================================================================
rem  Install zello-link and its dependencies on Windows.
rem
rem    install.bat            full install, including hardware extras
rem    install.bat --dev      also install test tooling
rem
rem  PortAudio ships inside the sounddevice wheel on Windows, so audio
rem  works out of the box. libopus does NOT -- see the note at the end.
rem =====================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv"
set "EXTRAS=hardware"
if /i "%~1"=="--dev" set "EXTRAS=hardware,dev"

echo.
echo === Checking Python ===

set "PYTHON="
for %%P in (py python) do (
    if not defined PYTHON (
        %%P -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
        if !errorlevel! equ 0 set "PYTHON=%%P"
    )
)

if not defined PYTHON (
    echo   FAIL  Python 3.11 or newer was not found on PATH.
    echo.
    echo   Install it from https://www.python.org/downloads/
    echo   Tick "Add python.exe to PATH" in the installer.
    exit /b 1
)
for /f "tokens=*" %%V in ('%PYTHON% --version') do echo   ok    %%V

echo.
echo === Creating virtual environment: %VENV% ===
if exist "%VENV%\Scripts\python.exe" (
    echo   ok    reusing existing %VENV%
) else (
    %PYTHON% -m venv "%VENV%"
    if !errorlevel! neq 0 (
        echo   FAIL  could not create the virtual environment
        exit /b 1
    )
    echo   ok    created %VENV%
)

"%VENV%\Scripts\python.exe" -m pip install --quiet --upgrade pip
echo   ok    pip up to date

echo.
echo === Installing zello-link[%EXTRAS%] ===
"%VENV%\Scripts\python.exe" -m pip install --quiet -e ".[%EXTRAS%]"
if !errorlevel! neq 0 (
    echo   FAIL  pip install failed
    echo.
    echo   If a hardware extra failed to build, install without it:
    echo     "%VENV%\Scripts\pip.exe" install -e .
    echo   Audio and PTT will be unavailable until it is installed.
    exit /b 1
)
echo   ok    installed

echo.
echo === Verifying ===
"%VENV%\Scripts\python.exe" -c "import importlib;^
 checks=[('package imports',lambda: importlib.import_module('zello_link')),^
 ('numpy',lambda: importlib.import_module('numpy')),^
 ('websockets',lambda: importlib.import_module('websockets')),^
 ('PortAudio (audio I/O)',lambda: __import__('sounddevice').query_devices()),^
 ('pyserial (serial PTT)',lambda: importlib.import_module('serial')),^
 ('hidapi (HID PTT/COS)',lambda: importlib.import_module('hid'))];^
 [print('  ok    '+n) if (lambda f: (f(), True)[1])(f) else None for n,f in checks]" 2>nul
if !errorlevel! neq 0 echo   warn  some optional components are unavailable

"%VENV%\Scripts\python.exe" -c "from zello_link.zello.opus import load_libopus; load_libopus(); print('  ok    libopus (codec)')" 2>nul
if !errorlevel! neq 0 (
    echo   warn  libopus NOT found - the bridge cannot encode or decode audio
    echo.
    echo         Windows has no package manager for this. Obtain opus.dll
    echo         ^(64-bit, matching your Python^) and place it either beside
    echo         the venv's python.exe or anywhere on PATH.
    echo         Prebuilt binaries: https://opus-codec.org/downloads/
)

echo.
echo === Done ===
echo   Next steps:
echo.
echo     1. Copy the example config and edit it:
echo          copy examples\bridge.yaml my-bridge.yaml
echo.
echo     2. Pick your audio devices interactively:
echo          .venv\Scripts\zello-link --config my-bridge.yaml --list-audio-devices
echo.
echo     3. Check the config without connecting or keying the radio:
echo          .venv\Scripts\zello-link --config my-bridge.yaml --validate
echo.
echo     4. Set the receive level, watching the live meter:
echo          .venv\Scripts\zello-link --config my-bridge.yaml --cos-monitor
echo.
echo   Secrets belong in the environment, not the config file:
echo     set ZELLO_AUTH_TOKEN=...
echo     set ZELLO_PASSWORD=...
echo.
echo   Note: PTT on Windows uses a COM port ^(e.g. ptt.tty_device: "COM3"^).
echo   Check Device Manager for the AIOC's assigned port number.
endlocal
