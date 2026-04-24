"""
Isolate the Qwen 3.6 crash — try several loading paths to find which step
explodes, with full Windows error reporting enabled.

Runs on the gaming PC. Writes results to C:/training/crash_report.txt
"""
import os, sys, traceback, json, time, faulthandler
from pathlib import Path

# Enable Python faulthandler so segfaults dump a trace before dying
faulthandler.enable()

REPORT = Path(r"C:/training/crash_report.txt")
REPORT.parent.mkdir(parents=True, exist_ok=True)

def log(s: str):
    msg = f"[{time.strftime('%H:%M:%S')}] {s}\n"
    print(msg, end="")
    sys.stdout.flush()
    with open(REPORT, "a") as f:
        f.write(msg)

log("=" * 70)
log("QWEN 3.6 CRASH INVESTIGATION")
log("=" * 70)

# Step 1 — versions
log("--- versions ---")
import platform; log(f"python: {sys.version.split()[0]}")
log(f"os: {platform.platform()}")
import torch; log(f"torch: {torch.__version__}")
log(f"cuda_available: {torch.cuda.is_available()}")
log(f"cuda_version: {torch.version.cuda}")
try:
    import unsloth; log(f"unsloth: {unsloth.__version__}")
except Exception as e: log(f"unsloth: IMPORT FAILED {e}")
try:
    import transformers; log(f"transformers: {transformers.__version__}")
except Exception as e: log(f"transformers: {e}")
try:
    import bitsandbytes as bnb; log(f"bitsandbytes: {bnb.__version__}")
except Exception as e: log(f"bitsandbytes: {e}")

# Step 2 — basic CUDA
log("--- step 2: basic CUDA ---")
try:
    x = torch.randn(10, device="cuda")
    y = x @ x.t()
    log(f"cuda matmul OK, device={x.device}, result.sum={y.sum().item():.3f}")
except Exception as e:
    log(f"CUDA BASIC FAILED: {e}")
    traceback.print_exc(file=open(REPORT, "a"))
    sys.exit(1)

# Step 3 — can we load a TINY Qwen model to confirm the pipeline works?
log("--- step 3: load tiny Qwen model (sanity) ---")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    log(f"qwen2.5-0.5B loaded OK on CUDA")
    del model; torch.cuda.empty_cache()
except Exception as e:
    log(f"qwen2.5-0.5B FAILED: {e}")
    traceback.print_exc(file=open(REPORT, "a"))

# Step 4 — try loading Qwen 3.6-27B in bf16 WITHOUT quantization, on CPU first
# This tells us if the CRASH is the model architecture or the bnb quantizer.
# bf16 27B weights = ~54 GB on disk but we don't need to load to GPU to prove
# the architecture parses. Use low_cpu_mem_usage and accelerate's empty_init.
log("--- step 4: meta-tensor init of Qwen 3.6-27B (architecture only, no weights) ---")
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B")
    log(f"config loaded: model_type={config.model_type} arch={config.architectures}")
    # Empty init — doesn't load weights, just instantiates the graph
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"architecture instantiated OK — {n_params:,} params")
    del model
except Exception as e:
    log(f"ARCHITECTURE FAILED: {e}")
    traceback.print_exc(file=open(REPORT, "a"))

# Step 5 — the failing step: Unsloth 4-bit load of Qwen 3.6-27B
log("--- step 5: Unsloth 4-bit load of Qwen 3.6-27B (THE FAILING STEP) ---")
try:
    import unsloth  # noqa: must come first
    from unsloth import FastLanguageModel
    log("importing unsloth OK")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3.6-27B",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    log("QWEN 3.6 4-BIT LOAD SUCCEEDED !!!")
    del model; torch.cuda.empty_cache()
except Exception as e:
    log(f"UNSLOTH 4-BIT FAILED: {type(e).__name__}: {e}")
    traceback.print_exc(file=open(REPORT, "a"))

# Step 6 — alternative: try transformers+bitsandbytes without Unsloth
log("--- step 6: transformers+bitsandbytes (no Unsloth) ---")
try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B", quantization_config=qc, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    log("TRANSFORMERS 4-BIT LOAD SUCCEEDED!")
    del model; torch.cuda.empty_cache()
except Exception as e:
    log(f"TRANSFORMERS 4-BIT FAILED: {type(e).__name__}: {e}")
    traceback.print_exc(file=open(REPORT, "a"))

# Step 7 — sanity: re-verify Qwen 2.5-14B still loads clean (our proven path)
log("--- step 7: re-verify Qwen 2.5-14B (known-good) ---")
try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    log("QWEN 2.5-14B 4-BIT LOAD OK — fallback path confirmed")
    del model
except Exception as e:
    log(f"QWEN 2.5-14B FAILED: {e}")

log("=" * 70)
log("INVESTIGATION COMPLETE")
log(f"full report: {REPORT}")
