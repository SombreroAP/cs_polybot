"""
Merge LoRA adapter into base, save as HuggingFace safetensors, register with
Ollama. Skips the llama.cpp compile step that blocks on Windows.

Newer Ollama (>=0.3.0) loads HF SafeTensors directly via FROM <directory>.
"""
import unsloth  # must come first

from unsloth import FastLanguageModel
import subprocess
from pathlib import Path

BASE = Path(r"C:\training")
ADAPTER = BASE / "smoketest_lora"
MERGED = BASE / "smoketest_merged"
MODEL_NAME = "smoketest-cs2-qwen05b"


def main():
    print("=" * 70)
    print("EXPORT: LoRA -> merged HF safetensors -> Ollama")
    print("=" * 70)

    print(f"\n[1/3] reload + merge adapter from {ADAPTER}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print("  -> reloaded")

    print(f"\n[2/3] save merged 16-bit HF model -> {MERGED}")
    model.save_pretrained_merged(str(MERGED), tokenizer, save_method="merged_16bit")
    print("  -> merged saved")

    print(f"\n[3/3] register with Ollama as '{MODEL_NAME}'")
    modelfile_path = BASE / "smoketest.Modelfile"
    # Use forward slashes for Ollama on Windows
    model_path_for_ollama = str(MERGED).replace("\\", "/")
    modelfile = f"""FROM {model_path_for_ollama}

SYSTEM \"\"\"CS2 Polymarket trader. Output ONE line of JSON, nothing else:
{{\"action\":\"buy\"|\"skip\",\"confidence\":0-1,\"reason\":\"short\"}}\"\"\"

PARAMETER temperature 0
PARAMETER num_predict 100
"""
    modelfile_path.write_text(modelfile)
    print(f"  -> Modelfile at {modelfile_path}")

    # Older Ollama may not support HF dir via FROM; newer does.
    result = subprocess.run(
        ["ollama", "create", MODEL_NAME, "-f", str(modelfile_path)],
        capture_output=True, text=True, timeout=600,
    )
    print("  stdout:", (result.stdout or "")[-400:])
    if result.returncode != 0:
        print("  stderr:", (result.stderr or "")[:400])
        return

    # Test inference
    print(f"\n[bonus] test inference via ollama run")
    test = subprocess.run(
        ["ollama", "run", MODEL_NAME,
         "GAME: CS2 FaZe vs Vitality  Series: 1-0  Alive: 5v3\nTrigger: round_end for FaZe\nBUY FaZe?"],
        capture_output=True, text=True, timeout=60,
    )
    print(f"  response: {test.stdout[:300]}")

    print("\n" + "=" * 70)
    print(f"EXPORT OK. Local test:  ollama run {MODEL_NAME}")
    print(f"Remote test:  curl http://localhost:11434/api/generate -d ...")
    print("=" * 70)


if __name__ == "__main__":
    main()
