@echo off
REM Save the current directory
pushd %~dp0

REM Check if main_cpu.py exists
if not exist main_cpu.py (
    echo Error: main_cpu.py not found in the current directory.
    popd
    pause
    exit /b 1
)

REM Run the Python script and check for errors
echo Running main_cpu.py...
python main_cpu.py
if %errorlevel% neq 0 (
    echo Error: main_cpu.py did not run successfully. Error level: %errorlevel%
    popd
    pause
    exit /b %errorlevel%
)

REM Provide success feedback
echo main_cpu.py ran successfully.

REM Return to the original directory
popd
pause
