@echo off

REM Create a virtual environment
echo Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Install the required dependencies
echo Installing dependencies...
pip install opencv-python
if %errorlevel% neq 0 (
    echo Failed to install opencv-python.
    pause
    exit /b 1
)

pip install torch
if %errorlevel% neq 0 (
    echo Failed to install torch.
    pause
    exit /b 1
)

pip install torchvision
if %errorlevel% neq 0 (
    echo Failed to install torchvision.
    pause
    exit /b 1
)

pip install torchaudio
if %errorlevel% neq 0 (
    echo Failed to install torchaudio.
    pause
    exit /b 1
)

pip install onnxruntime
if %errorlevel% neq 0 (
    echo Failed to install onnxruntime.
    pause
    exit /b 1
)

pip install pandas
if %errorlevel% neq 0 (
    echo Failed to install pandas.
    pause
    exit /b 1
)

pip install numpy
if %errorlevel% neq 0 (
    echo Failed to install numpy.
    pause
    exit /b 1
)

pip install gitpython==3.1.43
if %errorlevel% neq 0 (
    echo Failed to install gitpython.
    pause
    exit /b 1
)

pip install bettercam
if %errorlevel% neq 0 (
    echo Failed to install bettercam.
    pause
    exit /b 1
)

pip install colorama
if %errorlevel% neq 0 (
    echo Failed to install colorama.
    pause
    exit /b 1
)

pip install requests
if %errorlevel% neq 0 (
    echo Failed to install requests.
    pause
    exit /b 1
)

pip install ultralytics
if %errorlevel% neq 0 (
    echo Failed to install ultralytics.
    pause
    exit /b 1
)

pip install torch-directml
if %errorlevel% neq 0 (
    echo Failed to install torch-directml.
    pause
    exit /b 1
)

pip install onnxruntime-directml
if %errorlevel% neq 0 (
    echo Failed to install onnxruntime-directml.
    pause
    exit /b 1
)

REM Verify the installation
echo Listing installed packages...
pip list
if %errorlevel% neq 0 (
    echo Failed to list installed packages.
    pause
    exit /b 1
)

pause

REM Deactivate the virtual environment
deactivate

REM Exit the script
exit