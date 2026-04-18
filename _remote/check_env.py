"""Remote tool-check — runs on the gaming PC to report installed versions."""
import sys
print("python:", sys.version.split()[0])
mods = ["torch", "transformers", "trl", "unsloth", "datasets", "peft", "bitsandbytes", "accelerate"]
for m in mods:
    try:
        x = __import__(m)
        v = getattr(x, "__version__", "?")
        print(f"{m}: {v}")
    except ImportError as e:
        print(f"{m}: MISSING")

try:
    import torch
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        print(f"cuda_version: {torch.version.cuda}")
        print(f"torch_compiled_cuda: {torch.version.cuda}")
except Exception as e:
    print(f"cuda check failed: {e}")
