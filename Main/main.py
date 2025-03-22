import sys
import os
import time
import numpy as np
import cupy as cp
import cv2
import torch
import torchvision
import torchaudio
import torch_directml
import onnx
import onnxsim
import onnxruntime as ort
import onnxruntime_directml
import bettercam
import config
import customtkinter as ctk
import win32api
import win32con
import win32gui
from ultralytics import YOLO
from colorama import Fore, Style, init
from overlay import Overlay

# Only import if using TensorRT (.engine)
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


# ------------------ TensorRT Inference Class ------------------ #
class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        self.input_size = np.prod(self.input_shape) * np.float32().nbytes
        self.output_size = np.prod(self.output_shape) * np.float32().nbytes
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.bindings = [int(self.d_input), int(self.d_output)]

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        input_data = input_data.astype(np.float32).ravel()
        output_data = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_htod(self.d_input, input_data)
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(output_data, self.d_output)
        return output_data.reshape(self.output_shape)

# ------------------ BetterCam Enhanced ------------------ #
class BetterCamEnhanced:
    def __init__(self, max_buffer_len=config.maxBufferLen, target_fps=config.targetFPS, region=None, monitor_idx=0):
        self.camera = None
        self.max_buffer_len = max_buffer_len
        self.target_fps = target_fps
        self.region = region
        self.monitor_idx = monitor_idx
        self.is_capturing = False

    def start(self):
        self.camera = bettercam.create(monitor_idx=self.monitor_idx, max_buffer_len=self.max_buffer_len)
        self.camera.start(target_fps=self.target_fps)
        self.is_capturing = True

    def grab_frame(self):
        return self.camera.grab(region=self.region) if self.region else self.camera.grab()

    def stop(self):
        self.camera.stop()
        self.is_capturing = False

# ------------------ Load Model ------------------ #
def load_model():
    model_type = config.modelType.lower()

    if model_type == 'torch':
        print(Fore.CYAN + "[INFO] Loading PyTorch model using Ultralytics...")
        return YOLO(config.torchModelPath), 'torch'

    elif model_type == 'onnx':
        # Check GPU support order: CUDA (NVIDIA) > DirectML (AMD) > CPU
        available_providers = ort.get_available_providers()

        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider']
            print(Fore.GREEN + "[INFO] ONNX model will use CUDAExecutionProvider (NVIDIA GPU).")
        elif 'DmlExecutionProvider' in available_providers or 'DirectMLExecutionProvider' in available_providers:
            # Compatibility across versions
            providers = ['DmlExecutionProvider'] if 'DmlExecutionProvider' in available_providers else ['DirectMLExecutionProvider']
            print(Fore.YELLOW + "[INFO] ONNX model will use DirectMLExecutionProvider (AMD GPU).")
        else:
            providers = ['CPUExecutionProvider']
            print(Fore.RED + "[INFO] ONNX model will use CPUExecutionProvider (CPU only).")

        session = ort.InferenceSession(config.onnxModelPath, providers=providers)
        return session, 'onnx'

    elif model_type == 'engine':
        print(Fore.MAGENTA + "[INFO] Loading TensorRT engine...")
        return TensorRTInference(config.tensorrtModelPath), 'engine'

    else:
        raise ValueError(Fore.RED + "Unsupported modelType in config.py. Use 'torch', 'onnx', or 'engine'.")

# ------------------ Object Detection ------------------ #
def detect_objects(model, model_type, frame):
    if model_type == 'torch':
        return model.predict(source=frame, imgsz=(config.screenWidth, config.screenHeight), conf=config.confidenceThreshold, verbose=False)
    elif model_type == 'onnx':
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (config.screenWidth, config.screenHeight)).astype(np.float32)
        tensor = resized.transpose(2, 0, 1)[np.newaxis] / 255.0
        return model.run(None, {model.get_inputs()[0].name: tensor})
    elif model_type == 'engine':
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (config.screenWidth, config.screenHeight)).astype(np.float32)
        tensor = resized.transpose(2, 0, 1)[np.newaxis] / 255.0
        return model.infer(tensor)
    return None

# ------------------ Draw Bounding Boxes ------------------ #
def draw_boxes(frame, results, model_type):
    if results is None:
        return frame
    if model_type == 'torch':
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = box[:4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), config.boundingBoxColor, 2)
    elif model_type in ['onnx', 'engine']:
        for det in results[0]:
            if det[4] > config.confidenceThreshold:
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), config.boundingBoxColor, 2)
    return frame

def extract_boxes(results, model_type):
    boxes = []
    if model_type == 'torch':
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = box[:4]
                boxes.append([x1, y1, x2, y2])
    elif model_type in ['onnx', 'engine']:
        for det in results[0]:
            if det[4] > config.confidenceThreshold:
                x1, y1, x2, y2 = map(int, det[:4])
                boxes.append([x1, y1, x2, y2])
    return boxes

# ------------------ Main ------------------ #
def main():
    init(autoreset=True)
    input("Start your game and press Enter to continue...")

    camera = BetterCamEnhanced(target_fps=config.targetFPS, monitor_idx=config.monitorIdx)
    camera.start()

    model, model_type = load_model()
    overlay = Overlay(width=config.overlayWidth, height=config.overlayHeight, alpha=config.overlayAlpha)
    overlay.toggle()

    try:
        while True:
            frame = camera.grab_frame()
            if frame is None:
                continue

            results = detect_objects(model, model_type, frame)
            frame = draw_boxes(frame, results, model_type)
            boxes = extract_boxes(results, model_type)
            overlay.update(boxes)

            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        overlay.toggle()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
