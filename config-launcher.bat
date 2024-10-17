@echo off
REM Directly set the file path to config.py
set scriptPath="Paste your config.py path here"

REM Verify the entered file path exists
if not exist %scriptPath% (
    echo Error: The specified file does not exist. Please check the path.
    pause >nul
    exit /b 1
)

REM Open the config.py file in Notepad
echo Opening config.py in Notepad...
notepad %scriptPath%

REM Keep the window open
echo Press any key to exit...
pause >nul
