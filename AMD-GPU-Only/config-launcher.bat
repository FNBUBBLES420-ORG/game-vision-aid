@echo off
REM Save the current directory
pushd %~dp0

REM Check if config.py exists
if not exist config.py (
    echo Error: config.py not found.
    popd
    exit /b 1
)

REM Open the config.py file with the default text editor
echo Opening config.py...
start config.py

REM Provide success feedback
echo config.py opened successfully.

REM Return to the original directory
popd