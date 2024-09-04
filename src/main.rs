fn main() {
    // 🎮 GameVisionAid
    println!("GameVisionAid is an accessibility tool designed to assist visually impaired gamers by enhancing visual cues in video games.");
    println!("It uses real-time object detection to create customizable overlays around enemy players, making it easier to identify and engage with them during gameplay.\n");

    // 🚀 Features
    println!("🚀 Features:");
    println!("🖥️ Real-Time Screen Capture: Captures your screen in real-time using `mss` for fast and efficient screen capturing.");
    println!("🎯 Object Detection: Utilizes YOLOv5 for detecting enemies in video games.");
    println!("🟩 Customizable Overlays: Allows users to choose the color of the overlay boxes around detected enemies.");
    println!("🛠️ GPU Acceleration: Supports GPU acceleration for faster processing with CUDA-enabled GPUs.\n");

    // 🖥️ System Requirements
    println!("🖥️ System Requirements:");
    println!("- Operating System: Windows, Linux, or macOS");
    println!("- Python Version: [Python 3.11.6](https://github.com/KernFerm/Py3.11.6installer)");
    println!("- Hardware:");
    println!("  - CPU: Multi-core processor (Intel i5 or equivalent)");
    println!("  - GPU (Optional, but recommended for better performance): NVIDIA GPU with CUDA support");
    println!("  - RAM: 8 GB or more\n");

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
    println!("   pip install gitpython>=5.0.1");
    println!("5. Install PyTorch:");
    println!("   pip install torch==2.0.1 torchvision==0.15.2\n");

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

    // Python Code Explanation
    println!("Python Code Explanation:");
    println!("The following Python code snippet is part of the main functionality of the Game Vision Aid tool:");
    println!("print(\"Screen captured.\");");
    println!("results = detect_objects(model, model_type, frame, device);");
    println!("print(\"Objects detected.\");");
    println!("frame_with_boxes = draw_bounding_boxes(frame, results, overlay_color, model_type);");
    println!("print(\"Bounding boxes drawn.\");");
    println!("cv2.imshow(\"Overlay\", frame_with_boxes);");
    println!("if cv2.waitKey(1) & 0xFF == ord(\"q\"):");
    println!("   break;");
    println!("except KeyboardInterrupt:");
    println!("   print(\"Exiting program...\");");
    println!("finally:");
    println!("   cv2.destroyAllWindows();");

    // Contributing
    println!("Contributing:");
    println!("Contributions are welcome! Please fork the repository and submit a pull request with your changes.\n");

    // License
    println!("License:");
    println!("This project is licensed under the MIT License. See the LICENSE file for more details.\n");

    // Contact
    println!("Contact:");
    println!("For any questions or issues, please open an issue on the GitHub repository or contact the maintainers directly.\n");

    // Additional line
    println!("Welcome to Game Vision Aid!");
}
