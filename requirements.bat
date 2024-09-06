@echo off
setlocal

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.11.6 or later.
    pause
    exit /b 1
)

:: Install each package separately
echo Installing opencv-python...
pip install opencv-python
if %errorlevel% neq 0 (
    echo Failed to install opencv-python
    pause
    exit /b 1
)

echo Installing torch==2.4.1+cu118...
pip install torch==2.4.1+cu118
if %errorlevel% neq 0 (
    echo Failed to install torch==2.4.1+cu118
    pause
    exit /b 1
)

echo Installing torchvision==0.19.1+cu118...
pip install torchvision==0.19.1+cu118
if %errorlevel% neq 0 (
    echo Failed to install torchvision==0.19.1+cu118

    pause
    exit /b 1
)

echo Installing torchaudio==2.4.1+cu118...
pip install torchaudio==2.4.1+cu118
if %errorlevel% neq 0 (
    echo Failed to install torchaudio==2.4.1+cu118
    pause
    exit /b 1
)

echo Installing onnxruntime...
pip install onnxruntime
if %errorlevel% neq 0 (
    echo Failed to install onnxruntime
    pause
    exit /b 1
)

echo Installing requests...
pip install requests
if %errorlevel% neq 0 (
    echo Failed to install requests
    pause
    exit /b 1
)

echo Installing pandas...
pip install pandas
if %errorlevel% neq 0 (
    echo Failed to install pandas
    pause
    exit /b 1
)

echo Installing numpy...
pip install numpy
if %errorlevel% neq 0 (
    echo Failed to install numpy
    pause
    exit /b 1
)

echo Installing gitpython==3.1.43...
pip install gitpython==3.1.43
if %errorlevel% neq 0 (
    echo Failed to install gitpython==3.1.43
    pause
    exit /b 1
)

echo Installing bettercam...
pip install bettercam
if %errorlevel% neq 0 (
    echo Failed to install bettercam
    pause
    exit /b 1
)

echo Installing colorama...
pip install colorama
if %errorlevel% neq 0 (
    echo Failed to install colorama
    pause
    exit /b 1
)

echo All packages installed successfully.
endlocal
pause
