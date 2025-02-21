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
import onnxruntime as ort  # Import DirectML backend for ONNX
import subprocess  # Import subprocess for installing packages

# Enhanced BetterCam Initialization with multiple device support, error handling, and live feed
class BetterCamEnhanced:
    def __init__(self, max_buffer_len=config.maxBufferLen, target_fps=config.targetFPS, region=None):
        self.camera = None
        self.max_buffer_len = max_buffer_len
        self.target_fps = target_fps
        self.region = region
        self.is_capturing = False
        self._frame_count = 0
        self._start_time = time.time()
        self.window_name = "BetterCam Live Feed"
        self._last_frame = None
        self._frame_retry_count = 0
        self._max_retries = 3
        self._init_time = 1.0  # Time to wait after initialization

    def start(self):
        """Start screen capture with proper initialization."""
        try:
            # Close any existing camera instance
            if self.camera is not None:
                self.stop()
            
            # Create new camera instance
            self.camera = bettercam.create(max_buffer_len=self.max_buffer_len)
            
            # Wait before starting capture
            time.sleep(self._init_time)
            
            # Start capture
            self.camera.start(target_fps=self.target_fps)
            self.is_capturing = True
            self._start_time = time.time()
            print(Fore.GREEN + f"BetterCam started with target FPS: {self.target_fps}")
            
            # Wait for first frame
            time.sleep(0.5)
            
        except Exception as e:
            print(Fore.RED + f"Error starting BetterCam: {e}")
            self.cleanup()
            sys.exit(1)

    def stop(self):
        """Stop screen capture safely."""
        try:
            self.is_capturing = False
            if self.camera is not None:
                try:
                    self.camera.stop()
                    time.sleep(0.5)  # Wait for camera to stop
                except Exception as e:
                    print(Fore.RED + f"Error stopping camera: {e}")
                finally:
                    self.camera = None
        except Exception as e:
            print(Fore.RED + f"Error in stop: {e}")

    def cleanup(self):
        """Clean up resources properly."""
        try:
            # Stop capturing
            self.stop()
            
            # Destroy windows
            try:
                cv2.destroyAllWindows()
                for i in range(5):  # Sometimes needed to properly close windows
                    cv2.waitKey(1)
            except Exception as e:
                print(Fore.RED + f"Error closing windows: {e}")
            
            # Print statistics
            if self._frame_count > 0:
                elapsed_time = time.time() - self._start_time
                average_fps = self._frame_count / elapsed_time
                print(f"Average FPS: {average_fps:.1f}")
                
        except Exception as e:
            print(Fore.RED + f"Error during cleanup: {e}")
        finally:
            self.is_capturing = False
            self.camera = None

    def grab_frame(self):
        """Grab frame with improved error handling."""
        if not self.is_capturing or self.camera is None:
            return self._last_frame

        try:
            frame = None
            retry_count = 0
            
            while frame is None and retry_count < self._max_retries:
                try:
                    frame = self.camera.grab()
                    if frame is not None:
                        self._frame_count += 1
                        self._last_frame = frame
                        return frame
                except Exception as e:
                    retry_count += 1
                    if retry_count >= self._max_retries:
                        print(Fore.RED + f"Frame grab error after {retry_count} retries: {e}")
                    time.sleep(0.1)
            
            return self._last_frame
            
        except Exception as e:
            print(Fore.RED + f"Error in grab_frame: {e}")
            return self._last_frame

    def show_live_feed(self, model=None, model_type=None, overlay_color=None, device=None):
        """Show live feed with object detection overlay."""
        try:
            while self.is_capturing:
                frame = self.grab_frame()
                if frame is not None:
                    try:
                        # Convert BGRA to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        
                        # Perform detection if model is provided
                        if model is not None:
                            # Convert frame to RGB for PyTorch model
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            
                            # Resize frame to match YOLOv5's expected input size (640x640)
                            input_size = 640
                            original_height, original_width = frame_rgb.shape[:2]
                            
                            # Calculate resize scale while maintaining aspect ratio
                            scale = min(input_size / original_width, input_size / original_height)
                            new_width = int(original_width * scale)
                            new_height = int(original_height * scale)
                            
                            # Resize image
                            resized = cv2.resize(frame_rgb, (new_width, new_height))
                            
                            # Create square image with padding
                            square_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
                            dx = (input_size - new_width) // 2
                            dy = (input_size - new_height) // 2
                            square_img[dy:dy+new_height, dx:dx+new_width] = resized
                            
                            # Convert to tensor with proper memory format
                            frame_tensor = torch.from_numpy(square_img.copy()).to(device)
                            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
                            frame_tensor = frame_tensor.contiguous()  # Ensure tensor is contiguous
                            
                            # Perform inference
                            with torch.no_grad():
                                try:
                                    results = model(frame_tensor)
                                    
                                    # Store scale and padding info
                                    scale_info = {
                                        'scale': scale,
                                        'pad': (dx, dy),
                                        'original_size': (original_width, original_height)
                                    }
                                    
                                    # Draw detections
                                    frame_bgr = self.draw_detections(frame_bgr, results, overlay_color, scale_info)
                                except Exception as e:
                                    print(Fore.RED + f"Inference error: {e}")

                        # Display the frame
                        cv2.imshow(self.window_name, frame_bgr)

                    except Exception as e:
                        print(Fore.RED + f"Frame processing error: {e}")

                # Check for 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(Fore.YELLOW + "Exiting live feed...")
                    break

                # Small delay to prevent high CPU usage
                time.sleep(0.001)

        except KeyboardInterrupt:
            print(Fore.YELLOW + "Live feed interrupted.")
        finally:
            self.cleanup()

    def draw_detections(self, frame, results, color, scale_info):
        """Draw detection boxes for relevant classes."""
        try:
            # Classes we want to detect
            target_classes = {
                0,  # person
                14, # bird
                15, # cat
                16, # dog
                17, # horse
                18, # sheep
                19, # cow
                20, # elephant
                21, # bear
                22, # zebra
                23, # giraffe
            }

            confidence_threshold = 0.4  # Adjust this threshold as needed

            # Get scaling info
            scale = scale_info['scale']
            pad_x, pad_y = scale_info['pad']
            original_width, original_height = scale_info['original_size']

            # Process detections - handle YOLOv5 results format
            if hasattr(results, 'pred') and len(results.pred) > 0:
                detections = results.pred[0].cpu().numpy()
            elif isinstance(results, (list, tuple)) and len(results) > 0:
                detections = results[0].cpu().numpy()
            else:
                return frame  # Return original frame if no detections

            for detection in detections:
                if len(detection) >= 6:  # Check if detection has all required values
                    confidence = float(detection[4])
                    class_id = int(detection[5])

                    if class_id in target_classes and confidence > confidence_threshold:
                        # Get coordinates and scale back to original size
                        try:
                            x1 = int((float(detection[0]) - pad_x) / scale)
                            y1 = int((float(detection[1]) - pad_y) / scale)
                            x2 = int((float(detection[2]) - pad_x) / scale)
                            y2 = int((float(detection[3]) - pad_y) / scale)
                            
                            # Ensure coordinates are within frame bounds
                            x1 = max(0, min(x1, original_width - 1))
                            y1 = max(0, min(y1, original_height - 1))
                            x2 = max(0, min(x2, original_width - 1))
                            y2 = max(0, min(y2, original_height - 1))
                            
                            # Draw box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add label with confidence
                            class_names = {
                                0: 'person',
                                14: 'bird',
                                15: 'cat',
                                16: 'dog',
                                17: 'horse',
                                18: 'sheep',
                                19: 'cow',
                                20: 'elephant',
                                21: 'bear',
                                22: 'zebra',
                                23: 'giraffe'
                            }
                            class_name = class_names.get(class_id, 'unknown')
                            label = f'{class_name} {confidence:.2f}'
                            y1 = max(y1 - 10, 0)  # Ensure label is within frame
                            cv2.putText(frame, label, (x1, y1), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception as e:
                            print(Fore.RED + f"Error processing detection: {e}")
                            continue

            return frame
        except Exception as e:
            print(Fore.RED + f"Error drawing detections: {e}")
            return frame

def setup_device():
    """Set up and verify the compute device (AMD GPU or CPU)."""
    try:
        # First try to import torch_directml
        try:
            import torch_directml
            print(Fore.CYAN + "Initializing DirectML device...")
            dml = torch_directml.device()
            # Test the device with a small tensor operation
            test_tensor = torch.randn(1, 3, 64, 64).to(dml)
            test_result = test_tensor * 2
            print(Fore.GREEN + "DirectML device test successful")
            return dml
        except ImportError:
            print(Fore.YELLOW + "torch_directml not found, installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-directml"])
            import torch_directml
            return torch_directml.device()

    except Exception as e:
        print(Fore.YELLOW + f"GPU initialization error: {e}")
        print(Fore.YELLOW + "Falling back to CPU...")
        return torch.device('cpu')

def load_model(device):
    """
    Load the YOLOv5 model.
    :param device: Device to load the model on
    :return: Loaded YOLOv5 model or ONNX session.
    """
    try:
        print(Fore.WHITE + "Loading YOLOv5s model...")
        
        # Load model in eval mode
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()  # Set to evaluation mode
        
        # Move model to device before any inference
        if device.type != 'cpu':
            print(Fore.CYAN + f"Moving model to {device}...")
            try:
                # Convert model to float32
                model = model.float()
                model = model.to(device)
                print(Fore.GREEN + "Model successfully moved to device")
            except Exception as e:
                print(Fore.RED + f"Error moving model to device: {e}")
                print(Fore.YELLOW + "Falling back to CPU...")
                device = torch.device('cpu')
                model = model.to(device)
        
        # Print available classes
        if hasattr(model, 'names'):
            print(Fore.CYAN + "Available classes:")
            for idx, name in model.names.items():
                print(f"  {idx}: {name}")
        
        return model, 'torch', device
    except Exception as e:
        print(Fore.RED + f"Error loading model: {e}")
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
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # Save a frame occasionally for debugging
            if time.time() % 10 < 1:  # Save frame every ~10 seconds
                cv2.imwrite(f"debug_frame_{int(time.time())}.jpg", frame_bgr)
            return frame_bgr
        return None
    except Exception as e:
        print(Fore.RED + f"Error capturing screen: {e}")
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
            print(Fore.WHITE + f"PyTorch Detection Results: {results.xyxy[0].shape}")
        elif model_type == 'onnx':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (config.screenWidth, config.screenHeight))
            input_tensor = frame_resized.astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0).transpose(0, 3, 1, 2)
            input_tensor /= 255.0
            
            # Add debug prints
            print(Fore.WHITE + f"Input tensor shape: {input_tensor.shape}")
            print(Fore.WHITE + f"Input tensor range: {input_tensor.min():.3f} to {input_tensor.max():.3f}")
            
            outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
            results = outputs[0]
            print(Fore.WHITE + f"ONNX Detection Results shape: {results.shape}")
        return results
    except Exception as e:
        print(Fore.RED + f"Error during detection: {e}")
        print(Fore.RED + f"Full error details:", sys.exc_info())
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
            detections = 0
            for i, (xmin, ymin, xmax, ymax, conf, cls) in enumerate(results.xyxy[0]):
                print(Fore.WHITE + f"Detection {i}: confidence = {conf:.3f}")
                if conf > config.confidenceThreshold:
                    detections += 1
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            print(Fore.WHITE + f"Total detections above threshold: {detections}")
        elif model_type == 'onnx':
            detections = 0
            for i in range(len(results)):
                if len(results[i]) >= 5:
                    conf = results[i][4]
                    print(Fore.WHITE + f"Detection {i}: confidence = {conf:.3f}")
                    if conf > config.confidenceThreshold:
                        detections += 1
                        xmin, ymin, xmax, ymax = results[i][:4]
                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            print(Fore.WHITE + f"Total detections above threshold: {detections}")
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
    camera = None
    try:
        # Initialize colorama
        init(autoreset=True)
        
        # Ensure the game is running
        input(Fore.MAGENTA + "Make sure the game is running and visible on the screen. Press Enter to continue...")
        
        print(Fore.RED + "Any issues? Join our Discord Server: https://discord.fnbubbles420.org/invite")
        print(Fore.GREEN + "The program was created & developed by Bubbles The Dev for FNBUBBLES420 ORG.")

        # Set up device
        device = setup_device()
        
        if str(device) == 'cpu':
            print(Fore.YELLOW + "Warning: Running on CPU. This may be slower.")
            user_input = input("Do you want to continue anyway? (y/n): ").lower()
            if user_input != 'y':
                print(Fore.RED + "Exiting program...")
                sys.exit(0)

        # Initialize screen capture
        print(Fore.CYAN + "Initializing screen capture...")
        camera = BetterCamEnhanced(target_fps=config.targetFPS)
        camera.start()

        # Load model
        model, model_type, device = load_model(device)
        overlay_color = get_color_from_input()

        # Show feed
        print(Fore.CYAN + f"Running inference on device: {device}")
        camera.show_live_feed(model=model, model_type=model_type, overlay_color=overlay_color, device=device)

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nExiting program...")
    except Exception as e:
        print(Fore.RED + f"\nUnexpected error: {e}")
    finally:
        if camera is not None:
            camera.cleanup()
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
