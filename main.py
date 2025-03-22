import cv2
import numpy as np
import torch
import config
import time
from ultralytics import YOLO
import onnxruntime as ort
from overlay import Overlay
import bettercam
from colorama import Fore, init
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

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

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)
        self.output_shape = self.engine.get_binding_shape(1)
        self.input_size = np.prod(self.input_shape).item() * np.float32().nbytes
        self.output_size = np.prod(self.output_shape).item() * np.float32().nbytes
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

def load_model():
    model_type = config.modelType.lower()
    if model_type == 'torch':
        model = YOLO(config.torchModelPath)
    elif model_type == 'onnx':
        providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['DmlExecutionProvider']
        model = ort.InferenceSession(config.onnxModelPath, providers=providers)
    elif model_type == 'engine':
        model = TensorRTInference(config.tensorrtModelPath)
    else:
        raise ValueError("Invalid modelType in config.py")
    return model, model_type

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

def draw_boxes(frame, results, model_type):
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

def main():
    init(autoreset=True)
    input("Start your game. Press Enter to launch overlay and detection...")

    camera = BetterCamEnhanced(target_fps=config.targetFPS, monitor_idx=config.monitorIdx)
    camera.start()

    model, model_type = load_model()
    overlay = Overlay(config.overlayWidth, config.overlayHeight, config.overlayAlpha)
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
