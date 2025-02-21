@echo off
REM Save the current directory
pushd %~dp0

REM Check for GPU support first
echo Checking GPU support...
python check_gpu.py
if %errorlevel% neq 0 (
    echo Error: GPU check failed.
    pause
    popd
    exit /b %errorlevel%
)

REM Install required packages
echo Installing required packages...
python -m pip install torch torch-directml opencv-python colorama

REM Run the main script
echo Running main.py...
python main.py
if %errorlevel% neq 0 (
    echo Error: main.py did not run successfully.
    pause
    popd
    exit /b %errorlevel%
)

REM Provide success feedback
echo main.py ran successfully.

REM Return to the original directory
popd
pause