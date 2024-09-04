@echo off
setlocal

:: Check if NVIDIA GPU is present
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo NVIDIA GPU not detected. Please ensure you have an NVIDIA GPU installed.
    pause
    exit /b 1
)

:: Check if CUDA 11.8 is installed
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo CUDA 11.8 is not installed. Downloading CUDA 11.8 installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_win10.exe' -OutFile 'cuda_11.8.0_520.61.05_win10.exe'"
    echo Please run the downloaded installer to install CUDA 11.8.
    pause
    exit /b 1
)

:: Check if cuDNN 8.6.0 is installed
if not exist "%CUDA_PATH%\bin\cudnn64_8.dll" (
    echo cuDNN 8.6.0 is not installed. Downloading cuDNN 8.6.0...
    powershell -Command "Invoke-WebRequest -Uri 'https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.6.0/local_installers/11.8/cudnn-11.8-windows-x64-v8.6.0.163.zip' -OutFile 'cudnn-11.8-windows-x64-v8.6.0.163.zip'"
    echo Please extract the downloaded zip file and copy the contents to the CUDA installation directory.
    pause
    exit /b 1
)

echo CUDA 11.8 and cuDNN 8.6.0 are already installed.
endlocal
pause