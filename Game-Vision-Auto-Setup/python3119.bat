@echo off
echo This script will install Python 3.11.9 on your system.
echo Do you want to continue with the installation? (Y/N)

set /p choice="Enter your choice (Y/N): "
if /i "%choice%" neq "Y" (
    echo Installation cancelled by the user.
    pause
    exit /b 1
)

echo Downloading Python 3.11.9...
bitsadmin /transfer "PythonDownloadJob" /download /priority normal https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe "%cd%\python-3.11.9.exe"

:: Check if the download was successful
if not exist "%cd%\python-3.11.9.exe" (
    echo Download failed. Please check your internet connection or URL and try again.
    pause
    exit /b 2
)

echo Installing Python 3.11.9...
"%cd%\python-3.11.9.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0

:: Check the result of the installation
if %ERRORLEVEL% equ 0 (
    echo Python 3.11.9 has been installed successfully.
) else (
    echo Installation failed. Error code: %ERRORLEVEL%
    pause
    exit /b 3
)

echo Adding Python Scripts directory to user PATH...
:: Retrieve the current user PATH variable and append Python directories
for /f "tokens=2* delims=    " %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "userpath=%%b"
if not defined userpath set "userpath="

set "newuserpath=%userpath%;%LocalAppData%\Programs\Python\Python311\Scripts;%LocalAppData%\Programs\Python\Python311"

:: Use setx to update the user PATH permanently
setx PATH "%newuserpath%"

:: Update PATH for the current session
set PATH=%newuserpath%

echo Python Scripts directory has been added to the user PATH.
pause
