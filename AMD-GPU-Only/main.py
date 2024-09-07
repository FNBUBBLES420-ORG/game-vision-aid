import sys
import numpy as np  # Import NumPy for array manipulation
import cv2  # Import OpenCV for image processing
import torch  # Import PyTorch for YOLOv5 model
import onnxruntime as ort  # Import ONNX Runtime for ONNX model
import time  # Import time for time-related functions
import bettercam  # Import BetterCam for screen capture
import os  # Import os for file operations
from colorama import Fore, init  # Import colorama for colored text
import config  # Import config file for screen size and confidence threshold
import onnxruntime.directml as ort_dml  # Import DirectML backend for ONNX

# Enhanced BetterCam Initialization with multiple device support, error handling, and live feed
class BetterCamEnhanced:
    def __init__(self, max_buffer_len=config.maxBufferLen, target_fps=config.targetFPS, region=None):
        self.camera = None
        self.max_buffer_len = max_buffer_len
        self.target_fps = target_fps
        self.region = region
        self.is_capturing = False
        self.buffer = []

    def start(self):
        try:
            self.camera = bettercam.create(max_buffer_len=self.max_buffer_len)
            self.camera.start(target_fps=self.target_fps)
            self.is_capturing = True
            print(Fore.GREEN + f"BetterCam started with target FPS: {self.target_fps}")
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

    def show_live_feed(self):
        """Show live feed using OpenCV's imshow."""
        try:
            while self.is_capturing:
                frame = self.grab_frame()
                if frame is not None:
                    # Convert BGRA to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    # Display the live feed window
                    cv2.imshow("BetterCam Live Feed", frame_bgr)

                    # Exit if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(Fore.YELLOW + "Exiting live feed...")
                        break
                else:
                    print(Fore.YELLOW + "No frame to display.")
        except KeyboardInterrupt:
            print(Fore.YELLOW + "Live feed interrupted.")
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def stop(self):
        try:
            self.camera.stop()
            self.is_capturing = False
            print(Fore.GREEN + "BetterCam stopped.")
        except Exception as e:
            print(Fore.RED + f"Error stopping BetterCam: {e}")

def load_model(model_path=None):
    """
    Load the YOLOv5 model.
    :param model_path: Path to the custom model (.pt or .onnx) or model name from the YOLOv5 repository.
    :return: Loaded YOLOv5 model or ONNX session.
    """
    try:
        model_path = model_path or (config.torchModelPath if config.modelType == 'torch' else config.onnxModelPath)
        start_time = time.time()
        if model_path.endswith('.pt'):
            print(Fore.WHITE + "Loading PyTorch model...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            model_type = 'torch'
        elif model_path.endswith('.onnx'):
            print(Fore.WHITE + "Loading ONNX model...")
            model = ort_dml.InferenceSession(model_path, providers=['DmlExecutionProvider'])
            model_type = 'onnx'
        else:
            print(Fore.WHITE + "Loading YOLOv5 model from repository...")
            model = torch.hub.load('ultralytics/yolov5', model_path, force_reload=True)
            model_type = 'torch'
        end_time = time.time()
        print(Fore.WHITE + f"Model loaded in {end_time - start_time:.2f} seconds")
        return model, model_type
    except Exception as e:
        print(Fore.WHITE + f"Error loading model: {e}")
        sys.exit(1)

def capture_screen(camera, region=None):
    """
    Captures the screen using BetterCam.
    :param camera: The BetterCam camera instance.
    :param region: Optional region of the screen to capture.
    :return: Captured screen frame as a numpy array.
    """
    try:
        frame = camera.grab_frame()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR for OpenCV
            return frame_bgr
        return None
    except Exception as e:
        print(Fore.WHITE + f"Error capturing screen: {e}")
        return None

def detect_objects(model, model_type, frame, device):
    """
    Detects objects in the frame using YOLOv5 or ONNX model.
    :param model: YOLOv5 or ONNX model.
    :param model_type: Type of the model ('torch' or 'onnx').
    :param frame: The frame captured from the screen.
    :param device: Device to run the model on (CPU or GPU).
    :return: Results of detection.
    """
    try:
        if model_type == 'torch':
            frame_tensor = torch.from_numpy(frame).to(device)
            results = model(frame_tensor)
        elif model_type == 'onnx':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (config.screenWidth, config.screenHeight))  # Using config for screen size
            input_tensor = frame_resized.astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0).transpose(0, 3, 1, 2)
            input_tensor /= 255.0
            outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
            results = outputs[0]
        return results
    except Exception as e:
        print(Fore.WHITE + f"Error during detection: {e}")
        return None

def draw_bounding_boxes(frame, results, color, model_type):
    """
    Draws bounding boxes on the frame.
    :param frame: The frame to draw on.
    :param results: Detection results from YOLOv5 or ONNX model.
    :param color: Color of the bounding boxes.
    :param model_type: Type of the model ('torch' or 'onnx').
    :return: Frame with bounding boxes drawn.
    """
    if results is not None:
        if model_type == 'torch':
            for i, (xmin, ymin, xmax, ymax, conf, cls) in enumerate(results.xyxy[0]):
                if conf > config.confidenceThreshold:  # Apply confidence threshold from config
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        elif model_type == 'onnx':
            for i in range(len(results)):
                if len(results[i]) >= 5 and results[i][4] > config.confidenceThreshold:  # Apply confidence threshold from config
                    xmin, ymin, xmax, ymax, conf = results[i][:5]
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                else:
                    print(Fore.WHITE + f"Skipping result {i} due to insufficient values or low confidence.")
    return frame

def get_color_from_input():
    """
    Function to get a color from user input.
    :return: Selected color in BGR format.
    """
    color_dict = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (0, 165, 255),
        "purple": (128, 0, 128),
        "pink": (203, 192, 255),
        "brown": (42, 42, 165),
        "gray": (128, 128, 128),
        "light blue": (255, 127, 80),
        "light green": (50, 205, 50),
        "light red": (139, 0, 0),
        "light yellow": (255, 255, 224),
        "light cyan": (224, 255, 255),
        "light magenta": (255, 224, 255),
        "light orange": (255, 218, 185),
        "light purple": (229, 128, 255),
        "light pink": (255, 182, 193),
        "light brown": (165, 42, 42),
        "light gray": (211, 211, 211),
        "lime": (0, 255, 128),
        "turquoise": (64, 224, 208),
        "gold": (0, 215, 255),
        "silver": (192, 192, 192),
        "bronze": (140, 120, 83),
        "indigo": (75, 0, 130),
        "violet": (238, 130, 238),
        "teal": (0, 128, 128),
        "maroon": (128, 0, 0),
        "olive": (128, 128, 0),
        "navy": (0, 0, 128),
        "sky blue": (135, 206, 235),
        "lime green": (50, 205, 50),
        "dark red": (139, 0, 0),
        "dark yellow": (204, 204, 0),
        "dark cyan": (0, 139, 139),
        "dark magenta": (139, 0, 139),
        "dark orange": (255, 140, 0),
        "dark purple": (128, 0, 128),
        "dark pink": (199, 21, 133),
        "dark brown": (101, 67, 33),
        "dark gray": (169, 169, 169),
        "dark lime": (50, 205, 50),
        "dark turquoise": (0, 206, 209),
        "dark gold": (184, 134, 11),
        "dark silver": (169, 169, 169),
        "dark bronze": (140, 120, 83),
        "dark indigo": (75, 0, 130),
        "dark violet": (148, 0, 211),
        "dark teal": (0, 128, 128),
        "dark maroon": (128, 0, 0),
        "dark olive": (85, 107, 47),
        "dark navy": (0, 0, 128),
        "dark sky blue": (0, 191, 255),
        "metallic": (218, 165, 32),
        "neon": (255, 0, 255)
        
    }

    while True:
        color_input = input("Enter the overlay color (e.g., red, green, blue) or type 'exit' or 'q' to quit: ").strip().lower()
        if color_input in color_dict:
            return color_dict[color_input]
        elif color_input in ["exit", "q"]:
            print("Exiting program...")
            sys.exit(0)
        else:
            print("Invalid color name. Please try again.")

def main():
    # Initialize colorama
    init(autoreset=True)
    
    # Ensure the game is running and visible on the screen before starting the script
    input(Fore.MAGENTA + "Make sure the game is running and visible on the screen. Press Enter to continue...")
    
    print(Fore.RED + "Any issues? Join our Discord Server: https://discord.fnbubbles420.org/invite")
    print(Fore.GREEN + "The program was created & developed by Bubbles The Dev for FNBUBBLES420 ORG.")

    # Check for AMD GPU availability using DirectML on Windows
    try:
        device = torch.device('dml')  # Use DirectML on Windows for AMD GPUs
        print(Fore.CYAN + "DirectML-enabled AMD GPU found. Using DirectML backend.")
    except RuntimeError:
        device = 'cpu'
        print(Fore.YELLOW + "No supported GPU found. Using CPU.")


    # Set up BetterCam Enhanced
    camera = BetterCamEnhanced(target_fps=config.targetFPS)
    camera.start()

    # Load YOLOv5 or ONNX model
    model, model_type = load_model()
    if model_type == 'torch':
        model = model.to(device)

    # Get a color for the overlay from user input
    overlay_color = get_color_from_input()

    try:
        # Show live feed
        camera.show_live_feed()

    except KeyboardInterrupt:
        print(Fore.YELLOW + "Exiting program...")

    finally:
        # Clean up
        camera.stop()  # Stop BetterCam
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
