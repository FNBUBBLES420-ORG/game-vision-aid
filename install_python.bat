@echo off

REM Define the URL for the Python installer
set PYTHON_URL=https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe

REM Define the installer file name
set INSTALLER=python-3.11.6-amd64.exe

REM Download the Python installer
echo Downloading Python installer...
powershell -Command "Invoke-WebRequest -Uri %PYTHON_URL% -OutFile %INSTALLER%"

REM Check if the download was successful
if not exist %INSTALLER% (
    echo Failed to download the Python installer.
    exit /b 1
)

REM Run the installer with silent installation options
echo Installing Python...
%INSTALLER% /quiet InstallAllUsers=1 PrependPath=1

REM Check if the installation was successful
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python installation failed.
    exit /b 1
) else (
    echo Python installed successfully.
    python --version
)

REM Add Python to the system PATH
setx PATH "%PATH%;C:\Python311;C:\Python311\Scripts"

REM Clean up the installer file
del %INSTALLER%

pause

REM Exit the script
exit
