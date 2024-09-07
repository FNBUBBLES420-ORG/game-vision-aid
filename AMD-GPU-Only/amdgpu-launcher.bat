@echo off
REM Save the current directory
pushd %~dp0

REM Run the Python script and check for errors
echo Running main.py...
python main.py
if %errorlevel% neq 0 (
    echo Error: main.py did not run successfully.
    popd
    exit /b %errorlevel%
)

REM Provide success feedback
echo main.py ran successfully.

REM Return to the original directory
popd