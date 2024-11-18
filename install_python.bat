@echo off
echo This script will install Python 3.11.6 on your system.
echo Do you want to continue with the installation? (Y/N)

set /p choice="Enter your choice (Y/N): "
if /i "%choice%" neq "Y" (
    echo Installation cancelled by the user.
    pause
    exit /b 1
)

echo Downloading Python 3.11.6...
bitsadmin /transfer "PythonDownloadJob" /download /priority normal https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe "%cd%\python-3.11.6.exe"

:: Check if the download was successful
if not exist "%cd%\python-3.11.6.exe" (
    echo Download failed. Please check your internet connection or URL and try again.
    pause
    exit /b 2
)

echo Installing Python 3.11.6...
"%cd%\python-3.11.6.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

:: Check the result of the installation
if %ERRORLEVEL% equ 0 (
    echo Python 3.11.6 has been installed successfully.
) else (
    echo Installation failed. Error code: %ERRORLEVEL%
    pause
    exit /b 3
)

echo Adding Python Scripts directory to system PATH...
:: Retrieve the current PATH variable and append Python directories
set "newpath=%PATH%;C:\Programs\Python\Python311\Scripts;C:\Programs\Python\Python311"

:: Use setx to update the system PATH permanently
setx PATH "%newpath%" /M

:: Update PATH for the current session
set PATH=%newpath%

echo Python Scripts directory has been added to the system PATH.
pause
