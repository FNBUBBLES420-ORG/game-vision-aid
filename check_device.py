import torch
import platform
import importlib.util

print("\n🔍 Detecting Compute Device...\n")

# Check for NVIDIA (CUDA)
if torch.cuda.is_available():
    print("✅ CUDA is available (NVIDIA GPU)")
    print(f"🚀 CUDA Version : {torch.version.cuda}")
    print(f"🎮 GPU Name     : {torch.cuda.get_device_name(0)}")

# Check for AMD (DirectML)
elif importlib.util.find_spec("torch_directml"):
    try:
        import torch_directml
        dml_device = torch_directml.device()
        # Try allocating a dummy tensor to confirm
        test_tensor = torch.empty((1,)).to(dml_device)
        print("✅ DirectML is available (AMD GPU)")
        print(f"🖥️  DirectML Device: {dml_device}")
    except Exception as e:
        print("⚠️ Detected torch-directml, but could not initialize DirectML:", e)

# Fallback to CPU
else:
    print("⚠️ No GPU detected. Falling back to CPU execution.")
    print(f"🧠 CPU Info: {platform.processor() or platform.machine()}")

print("\n✅ Device check complete.")
