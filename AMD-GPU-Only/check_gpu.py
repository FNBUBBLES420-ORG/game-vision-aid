import torch
import sys
import os
from colorama import init, Fore

def get_gpu_details():
    """Get detailed GPU information."""
    try:
        import wmi
        import GPUtil
        from tabulate import tabulate
        
        # Get WMI GPU info
        w = wmi.WMI()
        gpu_info = []
        
        for gpu in w.Win32_VideoController():
            info = {
                "Name": gpu.Name,
                "Driver Version": gpu.DriverVersion,
                "Video Memory (GB)": round(int(gpu.AdapterRAM if gpu.AdapterRAM else 0) / (1024**3), 2),
                "Video Processor": gpu.VideoProcessor,
                "Video Memory Type": gpu.VideoMemoryType,
                "Current Resolution": f"{gpu.CurrentHorizontalResolution}x{gpu.CurrentVerticalResolution}",
                "Driver Date": gpu.DriverDate,
                "Status": gpu.Status,
                "DeviceID": gpu.DeviceID
            }
            gpu_info.append(info)
            
        # Try to get additional info using GPUtil
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                for info in gpu_info:
                    if gpu.name in info["Name"]:
                        info.update({
                            "Load": f"{gpu.load*100:.1f}%",
                            "Free Memory": f"{gpu.memoryFree:.1f} MB",
                            "Used Memory": f"{gpu.memoryUsed:.1f} MB",
                            "Total Memory": f"{gpu.memoryTotal:.1f} MB",
                            "Temperature": f"{gpu.temperature:.1f}°C",
                        })
        except Exception as e:
            print(Fore.YELLOW + f"Note: Some detailed GPU metrics unavailable: {e}")
            
        return gpu_info
    except Exception as e:
        print(Fore.RED + f"Error getting GPU details: {e}")
        return None

def check_gpu():
    init(autoreset=True)
    
    print(Fore.CYAN + "\n=== GPU Support Check ===")
    
    # Check Python version
    print(Fore.WHITE + f"\nPython version: {sys.version}")
    
    # Check PyTorch version
    print(Fore.WHITE + f"PyTorch version: {torch.__version__}")
    
    # Try importing torch_directml
    print(Fore.CYAN + "\n=== DirectML Support ===")
    try:
        import torch_directml
        print(Fore.GREEN + "✓ torch_directml is installed")
        dml = torch_directml.device()
        print(Fore.GREEN + f"✓ DirectML device: {dml}")
    except ImportError:
        print(Fore.YELLOW + "✗ torch_directml is not installed")
        print(Fore.WHITE + "Attempting to install torch_directml...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-directml"])
            print(Fore.GREEN + "✓ torch_directml installed successfully")
        except Exception as e:
            print(Fore.RED + f"✗ Error installing torch_directml: {e}")
    
    # Try using DirectML device
    try:
        device = torch.device('dml')
        test_tensor = torch.zeros((1,)).to(device)
        print(Fore.GREEN + "✓ DirectML device is working properly")
    except Exception as e:
        print(Fore.RED + f"✗ DirectML error: {e}")
    
    # Print system info
    print(Fore.CYAN + "\n=== System Information ===")
    try:
        import platform
        print(f"OS: {platform.system()} {platform.version()}")
        print(f"Machine: {platform.machine()}")
        print(f"Processor: {platform.processor()}")
        
        # Get and display detailed GPU information
        print(Fore.CYAN + "\n=== GPU Details ===")
        gpu_info = get_gpu_details()
        
        if gpu_info:
            for idx, gpu in enumerate(gpu_info, 1):
                print(Fore.GREEN + f"\nGPU {idx}:")
                for key, value in gpu.items():
                    print(f"{key}: {value}")
        
    except Exception as e:
        print(Fore.RED + f"Error getting system info: {e}")
    
    # Check for required packages
    print(Fore.CYAN + "\n=== Required Packages ===")
    required_packages = ['torch', 'torch_directml', 'opencv-python', 'numpy', 'colorama']
    for package in required_packages:
        try:
            __import__(package)
            print(Fore.GREEN + f"✓ {package} is installed")
        except ImportError:
            print(Fore.RED + f"✗ {package} is missing")

if __name__ == "__main__":
    try:
        # Install required packages for GPU checking
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wmi", "gputil", "tabulate"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
    except Exception as e:
        print(Fore.YELLOW + f"Note: Some GPU metrics may be unavailable: {e}")
    
    check_gpu()
    print("\nPress Enter to continue...")
    input() 