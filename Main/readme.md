# 🎯 Real-Time Object Detection Overlay with YOLO, TensorRT, and BetterCam

This project enables **real-time object detection** using **YOLOv5/YOLOv8**, with support for **PyTorch**, **ONNX**, and **TensorRT** models. Bounding boxes are rendered using a **transparent overlay** that displays on top of your game or any screen — perfect for assistive tools, AI research, and accessibility-focused applications.

---

## 🧠 Components

| File        | Description |
|-------------|-------------|
| `main.py`   | Core logic that captures frames, runs detection using the selected model, and displays bounding boxes via an overlay. |
| `config.py` | User-configurable settings: choose model type, set paths, screen dimensions, GPU support, overlay transparency, and more. |
| `overlay.py`| Creates a transparent, always-on-top, click-through overlay window that displays detection bounding boxes over your game or screen. |

---

## ⚙️ Features

✅ Supports **YOLOv5 / YOLOv8** via Ultralytics  
✅ Use **PyTorch (.pt)**, **ONNX (.onnx)**, or **TensorRT (.engine)** models  
✅ Automatic GPU detection: **NVIDIA (CUDA)**, **AMD (DirectML)**, or **CPU fallback**  
✅ Seamless integration with **BetterCam** for real-time screen capture  
✅ Transparent overlay using **Win32 API** (minimal FPS impact)  
✅ Multi-monitor support and full resolution control  
✅ Smooth, real-time bounding box rendering  
✅ Fully configurable through `config.py`

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install ultralytics opencv-python numpy torch torchvision torchaudio torch-directml onnx onnxruntime onnxruntime-directml onnx-simplifier pycuda tensorrt colorama customtkinter requests pandas cupy bettercam
```
- ⚠️ `BetterCam` must be installed or available in your environment. If it's a custom module, ensure it's in your Python path.

## 🧩 Configuration (config.py)
### ✨ Model Type

- Choose the model engine to use:
```bash
modelType = 'torch'  # Options: 'torch', 'onnx', 'engine'
```

## 🧠 Model Paths

- Point to your desired model files:

```bash
torchModelPath = 'models/yolov8n.pt'
onnxModelPath = 'models/yolov8n.onnx'
tensorrtModelPath = 'models/yolov8n.engine'
```

## 🎥 Screen & Overlay Settings

- Adjust `resolution`, `transparency`, and `monitor index`:

```bash
screenWidth = 640
screenHeight = 640
overlayWidth = 1920
overlayHeight = 1080
overlayAlpha = 200  # 0–255 (higher = more opaque)
```

## ⚡ GPU Support

- These values are used primarily for logic awareness — GPU detection is automatic:
```bash
useCuda = True         # CUDA (NVIDIA)
useDirectML = True     # DirectML (AMD)
```

## 🚀 Running the Program

- Once your configuration is set, launch the overlay:

```bash
python main.py
```

### You’ll be prompted to start your game. After pressing Enter:

- Your screen will begin real-time capture

- The selected model will run detection

- Bounding boxes will appear in the transparent overlay

- Press `Q` anytime to exit safely

📸 Supported Hardware

| GPU Type    | Engine Used                  |
|-------------|------------------------------|
| NVIDIA      | CUDA / TensorRT              |
| AMD Radeon  | DirectML (ONNX + PyTorch DML)|
| CPU Only    | ONNX CPU Execution           |


# ❤️ Credits
- Created with purpose by `Bubbles The Dev` 🫧
- Supporting accessible gaming, AI innovation, and empowering every player.
- Need extra features like FPS display, audio alerts, model switching GUI, or OBS integration?

### Hit me up — I got you! 😎



---
---
# 🚀 NVIDIA CUDA Installation Guide

### 1. **Download the NVIDIA CUDA Toolkit 11.8**

First, download the CUDA Toolkit 11.8 from the official NVIDIA website:

👉 [Nvidia CUDA Toolkit 11.8 - DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### 2. **Install the CUDA Toolkit**

- After downloading, open the installer (`.exe`) and follow the instructions provided by the installer.
- Make sure to select the following components during installation:
  - CUDA Toolkit
  - CUDA Samples
  - CUDA Documentation (optional)

### 3. **Verify the Installation**

- After the installation completes, open the `cmd.exe` terminal and run the following command to ensure that CUDA has been installed correctly:
  ```
  nvcc --version
  ```
This will display the installed CUDA version.

### **4. Install Cupy**
Run the following command in your terminal to install Cupy:
  ```
  pip install cupy-cuda11x
  ```

```
@echo off
echo MAKE SURE TO HAVE THE WHL DOWNLOADED BEFORE YOU CONTINUE!!!
pause
echo Click the link to download the WHL: press ctrl then left click with mouse
echo https://github.com/cupy/cupy/releases/download/v13.4.1/cupy_cuda11x-13.4.0-cp311-cp311-win_amd64.whl
pause

echo Installing CuPy from WHL...
pip install https://github.com/cupy/cupy/releases/download/v13.4.1/cupy_cuda11x-13.4.0-cp311-cp311-win_amd64.whl
pause

echo All packages installed successfully!
pause
```

## 5. CUDNN Installation 🧩
Download cuDNN (CUDA Deep Neural Network library) from the NVIDIA website:

👉 [Download CUDNN](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-windows-x86_64-8.9.6.50_cuda11-archive.zip/). (Requires an NVIDIA account – it's free).

## 6. Unzip and Relocate 📁➡️
Open the `.zip` cuDNN file and move all the folders/files to the location where the CUDA Toolkit is installed on your machine, typically:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```


## 7. Get TensorRT 8.6 GA 🔽
Download [TensorRT 8.6 GA](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip).

## 8. Unzip and Relocate 📁➡️
Open the `.zip` TensorRT file and move all the folders/files to the CUDA Toolkit folder, typically located at:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```


## 9. Python TensorRT Installation 🎡
Once all the files are copied, run the following command to install TensorRT for Python:

```
pip install "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python\tensorrt-8.6.1-cp311-none-win_amd64.whl"
```

🚨 **Note:** If this step doesn’t work, double-check that the `.whl` file matches your Python version (e.g., `cp311` is for Python 3.11). Just locate the correct `.whl` file in the `python` folder and replace the path accordingly.

## 10. Set Your Environment Variables 🌎
Add the following paths to your environment variables:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

# Setting Up CUDA 11.8 with cuDNN on Windows

Once you have CUDA 11.8 installed and cuDNN properly configured, you need to set up your environment via `cmd.exe` to ensure that the system uses the correct version of CUDA (especially if multiple CUDA versions are installed).

## Steps to Set Up CUDA 11.8 Using `cmd.exe`

### 1. Set the CUDA Path in `cmd.exe`

You need to add the CUDA 11.8 binaries to the environment variables in the current `cmd.exe` session.

Open `cmd.exe` and run the following commands:

```
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%
```
```
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp;%PATH%
```
```
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64;%PATH%
```
These commands add the CUDA 11.8 binary, lib, and CUPTI paths to your system's current session. Adjust the paths as necessary depending on your installation directory.

2. Verify the CUDA Version
After setting the paths, you can verify that your system is using CUDA 11.8 by running:
```
nvcc --version
```
This should display the details of CUDA 11.8. If it shows a different version, check the paths and ensure the proper version is set.

3. **Set the Environment Variables for a Persistent Session**
If you want to ensure CUDA 11.8 is used every time you open `cmd.exe`, you can add these paths to your system environment variables permanently:

1. Open `Control Panel` -> `System` -> `Advanced System Settings`.
Click on `Environment Variables`.
Under `System variables`, select `Path` and click `Edit`.
Add the following entries at the top of the list:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
```
This ensures that CUDA 11.8 is prioritized when running CUDA applications, even on systems with multiple CUDA versions.

4. **Set CUDA Environment Variables for cuDNN**
If you're using cuDNN, ensure the `cudnn64_8.dll` is also in your system path:
```
set PATH=C:\tools\cuda\bin;%PATH%
```
This should properly set up CUDA 11.8 to be used for your projects via `cmd.exe`.

#### Additional Information
- Ensure that your GPU drivers are up to date.
- You can check CUDA compatibility with other software (e.g., PyTorch or TensorFlow) by referring to their documentation for specific versions supported by CUDA 11.8.

```
import torch

print(torch.cuda.is_available())  # This will return True if CUDA is available
print(torch.version.cuda)  # This will print the CUDA version being used
print(torch.cuda.get_device_name(0))  # This will print the name of the GPU, e.g., 'NVIDIA GeForce RTX GPU Model'
```
run the `get_device.py` to see if you installed it correctly