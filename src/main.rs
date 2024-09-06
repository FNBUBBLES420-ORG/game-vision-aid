fn main() {
    // 🎮 GameVisionAid
    println!("GameVisionAid is an accessibility tool designed to assist visually impaired gamers by enhancing visual cues in video games.");
    println!("It uses real-time object detection to create customizable overlays around enemy players, making it easier to identify and engage with them during gameplay.\n");

    // 🚀 Features
    println!("🚀 Features:");
    println!("🖥️ Real-Time Screen Capture: Captures your screen in real-time using `bettercam` for fast and efficient screen capturing.");
    println!("🎯 Object Detection: Utilizes YOLOv5 for detecting enemies in video games.");
    println!("🟩 Customizable Overlays: Allows users to choose the color of the overlay boxes around detected enemies.");
    println!("🛠️ GPU Acceleration: Supports GPU acceleration for faster processing with CUDA-enabled GPUs.\n");
    println!("🎥 Live Feed Support: Displays a real-time live feed with object detection overlays.");

    // 🖥️ System Requirements
    println!("🖥️ System Requirements:");
    println!("- Operating System: Windows 10, 11");
    println!("- Python Version: [Python 3.11.6](https://github.com/KernFerm/Py3.11.6installer)");
    println!("- Hardware:");
    println!("  - CPU: Multi-core processor (Intel i5 or equivalent)");
    println!("  - GPU (Optional, but recommended for better performance): NVIDIA GPU with CUDA support");
    println!("  - RAM: 16 GB or more\n");

    // 📦 Installation
    println!("📦 Installation:");
    println!("### Prerequisites");
    println!("- NVIDIA GPU with CUDA support");
    println!("- CUDA Toolkit 11.8");
    println!("- cuDNN 8.9.7\n");

    println!("### Steps:");
    println!("1. Clone the repository from GitHub.");
    println!("2. Install CUDA and cuDNN:");
    println!("   - Follow the instructions to install CUDA 11.8 from the NVIDIA website.");
    println!("   - Download and install cuDNN 8.9.7 manually from the NVIDIA cuDNN download page.");
    println!("   - Ensure that the paths to CUDA and cuDNN are correctly set in your system environment variables.\n");

    println!("3. Install Python Dependencies:");
    println!("   pip install -r requirements.txt");
    println!("4. Install Additional Dependencies:");
    println!("   pip install gitpython==3.1.43");
    println!("5. Install PyTorch:");
    println!("   pip install torch==2.4.1 torchvision==0.19.1\n");
    
    // Supported Pytorch and Torchvision versions
    println!("torch==2.4.1");
    println!("torchvision==0.19.1\n");

    // Cuda and CuDNN versions
    println!("CUDA Version: 11.8");
    println!("cuDNN Version: 8.9.7\n");

    // Python Version
    println!("Python Version: 3.11.6\n");

    // Programming Languages Used
    println!("Programming Languages Used:");
    println!("1. Python\n");
    println!("2. JavaScript\n");
    println!("3. Rust\n");
    println!("4. Batchfile\n");
    println!("5. PowerShell\n");

    // Dependencies
    println!("Dependencies:");
    println!("1. PyTorch");
    println!("2. Torchvision");
    println!("3. OpenCV");
    println!("4. NumPy");
    println!("5. CUDA");
    println!("6. cuDNN\n");
    println!("7. GitPython\n");
    println!("8. BetterCAM\n");
    println!("9. Pandas\n");
    println!("10. Colorama\n");
    println!("11. OnnxRuntime\n");

    // Os Supported
    println!("OS Supported:");
    println!("1. Windows 10");
    println!("2. Windows 11\n");

    // Yolo version
    println!("YOLO Version: YOLOv5\n");
  
    // Usage
    println!("Usage:");
    println!("After building the project, you can run the executable to start using the Game Vision Aid tools:");
    println!("./target/release/game-vision-aid\n");

    // JavaScript Code Explanation
    println!("JavaScript Code Explanation:");
    println!("The following JavaScript code snippet provides instructions for manually downloading and installing cuDNN:");
    println!("console.log(\"cuDNN is not installed. Please follow these steps to download and install cuDNN 8.9.7 manually:\");");
    println!("console.log(\"1. Go to the NVIDIA cuDNN download page: https://developer.nvidia.com/rdp/cudnn-download\");");
    println!("console.log(\"2. Log in with your NVIDIA Developer account.\");");
    println!("console.log(\"3. Download the file 'cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip'.\");");
    println!("console.log(\"4. Extract the contents of the downloaded zip file.\");");
    println!("console.log(\"5. Copy the following files to the CUDA installation directory:\");");
    println!("console.log(\"   - Copy 'cudnn64_8.dll' to 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\bin'\");");
    println!("console.log(\"   - Copy 'cudnn.h' to 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\include'\");");
    println!("console.log(\"   - Copy 'cudnn.lib' to 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\lib\\\\x64'\");");
    println!("console.log(\"6. Press any key to continue once you have moved and unzipped the folder.\");");
    println!("console.log(\"Note: If you encounter issues, you might need to manually add the following paths to your system PATH:\");");
    println!("console.log(\"   - 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\bin'\");");
    println!("console.log(\"   - 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\libnvvp'\");");
    println!("console.log(\"   - 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\lib'\");");

    
    // Config.py
    println!("Config.py:");
    println!("# Configuration for BetterCam Screen Capture and YOLO model");
    println!("# Screen Capture Settings");
    println!("screenWidth = 480  # Updated screen width for better resolution");
    println!("screenHeight = 480  # Updated screen height for better resolution");
    println!("# Object Detection Settings");
    println!("confidenceThreshold = 0.5  # Confidence threshold for object detection");
    println!("nmsThreshold = 0.4  # Non-max suppression threshold to filter overlapping boxes");
    println!("# YOLO Model Settings");
    println!("modelType = 'onnx'  # Choose 'torch' or 'onnx' based on the model you want to load");
    println!("torchModelPath = 'models/fn_v5.pt'  # Path to YOLOv5 PyTorch model");
    println!("onnxModelPath = 'models/fn_v5.onnx'  # Path to YOLOv5 ONNX model");
    println!("# BetterCam Settings");
    println!("targetFPS = 60  # Frames per second for capturing");
    println!("maxBufferLen = 512  # Max buffer length for storing frames");
    println!("region = None  # Region for capture (set to None for full screen)");
    println!("useNvidiaGPU = True  # Set to True to enable GPU acceleration if available");
    println!("# Colors for Bounding Boxes");
    println!("boundingBoxColor = (0, 255, 0)  # Default bounding box color in BGR format");
    println!("highlightColor = (0, 0, 255)  # Color for highlighted objects");


    // Python Code Explanation
    println!("Python Code Explanation:");
    println!("The following Python code snippet provides instructions for installing the required Python dependencies:");
    println!("The Python code has been written in Python 3.11.6\n");

    // Contributing
    println!("Contributing:");
    println!("Contributions are welcome! Please fork the repository and submit a pull request with your changes.\n");

    // License
    println!("License:");
    println!("This project is proprietary and all rights are reserved by the author.");
    println!("Unauthorized copying, distribution, or modification of this project is strictly prohibited./n"); 
    println!("Unless You have written permission from the Developer or the FNBUBBLES420 ORG./n");

    // Contact
    println!("Contact:");
    println!("For any questions or issues, please open an issue on the GitHub repository or contact the maintainers directly.\n");

    // Additional line
    println!("Welcome to Game Vision Aid!");
}
