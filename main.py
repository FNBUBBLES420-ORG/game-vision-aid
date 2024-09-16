import sys
import numpy as np
import cupy as cp  # Importing CuPy for GPU-accelerated operations
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


# Overlay class from overlay2.py
class Overlay:
    def __init__(self, width=320, height=320, alpha=0.75):
        self.overlay = None
        self.canvas = None
        self.width = width
        self.height = height
        self.alpha = alpha  # Alpha level can be passed in

    def set_clickthrough(self, hwnd):
        try:
            styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, 0x00000001)
        except Exception as e:
            print(f"Error setting clickthrough: {e}")

    def toggle(self):
        if self.overlay is None:
            try:
                left = int(win32api.GetSystemMetrics(0) / 2 - self.width / 2)
                top = int(win32api.GetSystemMetrics(1) / 2 - self.height / 2)
                self.overlay = ctk.CTkToplevel()
                self.overlay.geometry(f"{self.width}x{self.height}+{left}+{top}")
                self.overlay.overrideredirect(True)
                self.overlay.config(bg='#000000')
                self.overlay.attributes("-alpha", self.alpha)
                self.overlay.wm_attributes("-topmost", 1)
                self.overlay.attributes('-transparentcolor', '#000000', '-topmost', 1)
                self.overlay.resizable(False, False)
                self.set_clickthrough(self.overlay.winfo_id())
                self.canvas = ctk.CTkCanvas(self.overlay, width=self.width, height=self.height, bg='black', highlightbackground='white')
                self.canvas.pack()
            except Exception as e:
                print(f"Error creating overlay: {e}")
        else:
            self.overlay.destroy()
            self.canvas.destroy()
            self.overlay = None
            self.canvas = None

    def update(self, coordinates):
        if self.overlay is not None:
            self.overlay.update()
            if self.canvas is not None:
                self.canvas.delete("all")
                if len(coordinates):
                    for coord in coordinates:
                        x_min, y_min, x_max, y_max = map(int, coord)
                        self.canvas.create_rectangle(x_min, y_min, x_max, y_max, outline="white", width=2)


def load_model(model_path=None):
    try:
        model_path = model_path or (config.torchModelPath if config.modelType == 'torch' else config.onnxModelPath)
        start_time = time.time()
        if model_path.endswith('.pt'):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            model_type = 'torch'
        elif model_path.endswith('.onnx'):
            model = ort.InferenceSession(model_path)
            model_type = 'onnx'
        else:
            model = torch.hub.load('ultralytics/yolov5', model_path, force_reload=True)
            model_type = 'torch'
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")
        return model, model_type
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def capture_screen(camera, region=None):
    try:
        frame = camera.grab_frame()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame_bgr
        return None
    except Exception as e:
        print(f"Error capturing screen: {e}")
        return None


def detect_objects(model, model_type, frame, device):
    try:
        if model_type == 'torch':
            # Use cupy for tensor conversion if necessary
            frame_tensor = torch.from_numpy(cp.asnumpy(frame)).to(device)
            results = model(frame_tensor)
        elif model_type == 'onnx':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (config.screenWidth, config.screenHeight))
            input_tensor = frame_resized.astype(cp.float32)
            input_tensor = cp.expand_dims(input_tensor, axis=0).transpose(0, 3, 1, 2)
            input_tensor /= 255.0
            outputs = model.run(None, {model.get_inputs()[0].name: cp.asnumpy(input_tensor)})
            results = outputs[0]
        return results
    except Exception as e:
        print(f"Error during detection: {e}")
        return None


def draw_bounding_boxes(frame, results, color, model_type):
    if results is not None:
        if model_type == 'torch':
            for i, (xmin, ymin, xmax, ymax, conf, cls) in enumerate(results.xyxy[0]):
                if conf > config.confidenceThreshold:
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        elif model_type == 'onnx':
            for i in range(len(results)):
                if len(results[i]) >= 5 and results[i][4] > config.confidenceThreshold:
                    xmin, ymin, xmax, ymax, conf = results[i][:5]
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    return frame


def get_color_from_input():
    color_dict = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
    }
    while True:
        color_input = input("Enter the overlay color (red, green, blue): ").strip().lower()
        if color_input in color_dict:
            return color_dict[color_input]
        else:
            print("Invalid color name. Please try again.")


def main():
    init(autoreset=True)
    input("Make sure the game is running. Press Enter to continue...")
    
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA-enabled GPU found. Using GPU.")
    else:
        device = 'cpu'
        print("No CUDA-enabled GPU found. Using CPU.")

    camera = BetterCamEnhanced(target_fps=config.targetFPS, monitor_idx=config.monitorIdx)
    camera.start()

    model, model_type = load_model()
    if model_type == 'torch':
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
                cv2.imshow("YOLOv5 Detection", frame)

                if results:
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
