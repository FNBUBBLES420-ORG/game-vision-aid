import sys
import numpy as np
import cupy as cp  # Importing CuPy for GPU-accelerated operations (NVIDIA GPU)
import cv2
import torch
import onnxruntime as ort
import time
import bettercam
import os
from colorama import Fore, Style, init
import config
import customtkinter as ctk
import win32api
import win32con
import win32gui

# Enhanced BetterCam Initialization with multiple device support, error handling
class BetterCamEnhanced:
    def __init__(self, max_buffer_len=config.maxBufferLen, target_fps=config.targetFPS, region=None, monitor_idx=0):
        self.camera = None
        self.max_buffer_len = max_buffer_len
        self.target_fps = target_fps
        self.region = region
        self.monitor_idx = monitor_idx  # Monitor index for multi-monitor support
        self.is_capturing = False
        self.buffer = []

    def start(self):
        try:
            # Create BetterCam instance for the selected monitor
            self.camera = bettercam.create(monitor_idx=self.monitor_idx, max_buffer_len=self.max_buffer_len)
            self.camera.start(target_fps=self.target_fps)
            self.is_capturing = True
            print(Fore.GREEN + f"BetterCam started on monitor {self.monitor_idx} with target FPS: {self.target_fps}")
        except Exception as e:
            print(Fore.RED + f"Error starting BetterCam: {e}")
            sys.exit(1)

    def grab_frame(self):
        try:
            if self.region:
                frame = self.camera.grab(region=self.region)
            else:
                frame = self.camera.grab()
            if frame is not None:
                return frame
            else:
                print(Fore.RED + "Failed to grab frame.")
                return None
        except Exception as e:
            print(Fore.RED + f"Error capturing frame: {e}")
            return None

    def stop(self):
        try:
            self.camera.stop()
            self.is_capturing = False
            print(Fore.GREEN + "BetterCam stopped.")
        except Exception as e:
            print(Fore.RED + f"Error stopping BetterCam: {e}")

# Model loading function with support for both YOLOv5 and YOLOv8, CUDA, and DirectML
def load_model(model_path=None):
    try:
        model_path = model_path or (config.torchModelPath if config.modelType == 'torch' else config.onnxModelPath)
        start_time = time.time()

        # Check for PyTorch model (.pt)
        if model_path.endswith('.pt'):
            if 'yolov8' in model_path.lower():
                model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, force_reload=True)
            else:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            model_type = 'torch'

        # Check for ONNX model (.onnx)
        elif model_path.endswith('.onnx'):
            # Try to load with DirectML if CUDA is not available
            if torch.cuda.is_available():
                providers = ['CUDAExecutionProvider']
            else:
                providers = ['DmlExecutionProvider']
            model = ort.InferenceSession(model_path, providers=providers)
            model_type = 'onnx'

        # Check for TensorRT model (.engine)
        elif model_path.endswith('.engine'):
            # TensorRT model loading logic should be added here
            print(Fore.YELLOW + "TensorRT model detected. Ensure the correct environment for TensorRT is set up.")
            model = None  # Placeholder for TensorRT engine loading
            model_type = 'engine'

        else:
            raise ValueError(f"Unsupported model format for {model_path}")

        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")
        return model, model_type
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# Object detection function with CUDA and DirectML support
def detect_objects(model, model_type, frame, device):
    try:
        if model_type == 'torch':
            # Use cupy for tensor conversion if necessary (CUDA)
            frame_tensor = torch.from_numpy(cp.asnumpy(frame)).to(device)
            results = model(frame_tensor)

        elif model_type == 'onnx':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (config.screenWidth, config.screenHeight))
            input_tensor = frame_resized.astype(np.float32)  # Removed cp for non-CUDA GPUs
            input_tensor = np.expand_dims(input_tensor, axis=0).transpose(0, 3, 1, 2)
            input_tensor /= 255.0

            # Use GPU-accelerated tensor processing with DirectML or CUDA
            outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})

            results = outputs[0]

        elif model_type == 'engine':
            print(Fore.YELLOW + "TensorRT inference is not yet implemented. Placeholder code for TensorRT.")
            results = None  # Placeholder for TensorRT inference results

        return results
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

# Main function that sets up GPU support for both NVIDIA and AMD
def main():
    init(autoreset=True)
    input("Make sure the game is running. Press Enter to continue...")

    # Determine if an NVIDIA (CUDA) or AMD (DirectML) GPU is available
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA-enabled GPU found. Using NVIDIA GPU.")
    else:
        try:
            # Check if DirectML for AMD GPUs is available
            device = torch.device('dml')  # DirectML for PyTorch on AMD GPUs
            print("Using AMD GPU with DirectML.")
        except:
            device = 'cpu'
            print("No CUDA or DirectML GPU found. Using CPU.")

    camera = BetterCamEnhanced(target_fps=config.targetFPS, monitor_idx=config.monitorIdx)
    camera.start()

    model, model_type = load_model()
    if model_type == 'torch' and device != 'cpu':
        model = model.to(device)

    overlay = Overlay(width=config.overlayWidth, height=config.overlayHeight, alpha=config.overlayAlpha)
    overlay.toggle()  # Start the overlay

    overlay_color = get_color_from_input()

    try:
        while True:
            frame = capture_screen(camera)
            if frame is not None:
                results = detect_objects(model, model_type, frame, device)
                frame = draw_bounding_boxes(frame, results, overlay_color, model_type)
                cv2.imshow("YOLO Detection", frame)

                if results and model_type == 'torch':
                    coordinates = [[xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax, _, _ in results.xyxy[0]]
                    overlay.update(coordinates)  # Update the overlay with bounding box coordinates

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        overlay.toggle()  # Close the overlay
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
