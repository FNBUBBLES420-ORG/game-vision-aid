# GAME-VISION-AID AUTO SETUP EXE

- `SHA256 3A687AD5C2E7765E2D3F4ACF6A3491FE144165E57A786D225C25D0C03BAC72CE`

## INSTALL PYTHON 

- Use the `python3119.bat` to install python.
- Then run the `autosetup.exe` it will install everything needed to run the application.
- For nvidia users, you may need to install some stuff manually.
- For amd users, you may need to install some stuff manually.

## 🔧 Installation Guide
### 1️⃣ Run the Script
- **Double-click** `python3119.bat`
- **Type `Y` and press Enter** to begin installation

### 2️⃣ Python Installation 🐍
If Python **is not installed**, the script will:
- Download & Install **Python 3.11.9**
- Automatically **add Python to PATH**

---

# AutoSetup Script

This repository contains a Python-based automation process that installs required dependencies and configures the environment based on your GPU type. It is designed for Windows systems and uses WMIC for GPU detection.

---

## ⚠️ Important First Step

1. **Make sure you have Python 3119 installed** by running `python3119.bat` **as a non-admin** user.  
2. **Then**, run the `autosetup.exe` to begin the automated setup process.

---

## Features

- **GPU Detection:**  
  Uses WMIC to detect if your system has an NVIDIA or AMD GPU. Defaults to CPU mode if neither is detected.

- **Manual Installation Prompts:**  
  - **NVIDIA Users:**  
    - Register for a free NVIDIA Developer account.
    - Install CUDA 11.8, cuDNN, TensorRT, and Visual Studio 2022 Community Edition (with C++ support).
  - **AMD Users:**  
    - Install Visual Studio 2022 Community Edition for DirectML support.

- **Environment Configuration:**  
  For NVIDIA GPUs, the script updates the `PATH` environment variable to include essential CUDA directories.

- **Dependency Installation:**  
  Installs GPU-specific and general Python packages:
  - **NVIDIA:**  
    - Installs CUDA-enabled versions of PyTorch, TorchVision, TorchAudio, and additional NVIDIA libraries.
  - **AMD:**  
    - Installs AMD-compatible libraries including support for `torch-directml`.
  - **CPU-only:**  
    - Installs CPU-compatible versions of the packages.
  - Additionally, installs a common set of Python packages required by your project.

---

## Prerequisites

- **Operating System:** Windows  
- **Python 3119:** Make sure you have installed it via `python3119.bat` (run as non-admin).
- **WMIC:** Used for GPU detection (deprecated on newer Windows versions, but still works on many systems).

---

## Getting Started

1. **Install Python 3119**  
   Run the `python3119.bat` file **as non-admin** to install Python 3119.

2. **Run AutoSetup**  
   Double-click or run `autosetup.exe` from the Command Prompt to start the automated process.

3. **Follow On-Screen Prompts**  
   - The script will detect your GPU type.
   - For NVIDIA or AMD, it will prompt you for manual installation steps (CUDA, cuDNN, Visual Studio, etc.).
   - Press **Enter** after each manual step.

---

## Installation Process

- **GPU-Specific Installs:**
  - If you have an **NVIDIA GPU**, the script will guide you through:
    1. Creating a free NVIDIA Developer account.
    2. Downloading/installing CUDA 11.8, cuDNN, TensorRT.
    3. Installing Visual Studio 2022 Community Edition (with C++).
  - If you have an **AMD GPU**, the script will guide you through:
    1. Installing Visual Studio 2022 Community Edition (with C++).
    2. Installing the AMD-compatible Python libraries.
  - If you have **no supported GPU**, it will install CPU-only libraries.

- **Environment Variable Setup:**  
  For NVIDIA GPUs, the script updates your `PATH` to include CUDA and related libraries.

- **Python Dependencies:**  
  Depending on your GPU, it installs the correct version of PyTorch, TorchVision, TorchAudio, and more.

---

## Completion

Once the script finishes:

- You will see a confirmation that Python 3119, dependencies, and any required GPU libraries have been installed.
- You can now proceed to use your environment for development, gaming, or any other tasks.

---

## Troubleshooting

- **WMIC Compatibility:**  
  If WMIC is unavailable or deprecated on your system, consider alternative GPU detection methods.
- **Installation Errors:**  
  Ensure you have followed all manual steps (CUDA, Visual Studio, etc.) and that you have the necessary permissions.


## Support the Project ⭐
If you find this project useful, please consider joining our Discord community, where we offer:

- Great community support
- Mental health support
- Development discussions

[Click Here To Join FNBubbles420 Org Non Profit Discord Server](https://discord.fnbubbles420.org/invite)


## LICENSE
This project is proprietary and all rights are reserved by the author.
Unauthorized copying, distribution, or modification of this project is strictly prohibited.
Unless you have written permission from the Developer or the FNBUBBLES420 ORG.

## Copyright Notice
© 2024 Bubbles The Dev and FNBUBBLES420ORG. All rights reserved.

This image, including its design, text, and visual elements, is protected under copyright law. Unauthorized use, reproduction, distribution, or modification without the express written permission of Bubbles The Dev and FNBUBBLES420ORG is prohibited. For licensing or usage inquiries, please contact media@fnbubbles420.org.
