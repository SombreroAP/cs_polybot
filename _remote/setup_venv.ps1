# Run on gaming PC. Creates a fresh CUDA-enabled venv for smoke-test training
# without disturbing the existing C:\training\venv.

$ErrorActionPreference = "Stop"
$python311 = "C:\Users\Andre\AppData\Local\Programs\Python\Python311\python.exe"
$target = "C:\training\venv_smoke"

Write-Host "=== 1. create venv at $target ===" -ForegroundColor Cyan
& $python311 -m venv $target
& "$target\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools

Write-Host "`n=== 2. install PyTorch CUDA 12.8 (Blackwell-ready) ===" -ForegroundColor Cyan
# cu128 nightly / stable for RTX 50-series
& "$target\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "`n=== 3. install Unsloth for cu128 ===" -ForegroundColor Cyan
& "$target\Scripts\python.exe" -m pip install "unsloth[cu128] @ git+https://github.com/unslothai/unsloth"

Write-Host "`n=== 4. install rest of training deps ===" -ForegroundColor Cyan
& "$target\Scripts\python.exe" -m pip install datasets trl transformers peft bitsandbytes accelerate sentencepiece protobuf

Write-Host "`n=== 5. verify ===" -ForegroundColor Cyan
& "$target\Scripts\python.exe" -c @"
import torch, transformers, trl, unsloth, datasets, peft
print('python      :', __import__('sys').version.split()[0])
print('torch       :', torch.__version__)
print('cuda        :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device      :', torch.cuda.get_device_name(0))
    print('cuda_v      :', torch.version.cuda)
print('transformers:', transformers.__version__)
print('trl         :', trl.__version__)
print('unsloth     :', unsloth.__version__)
print('datasets    :', datasets.__version__)
print('peft        :', peft.__version__)
"@

Write-Host "`n=== DONE ===" -ForegroundColor Green
