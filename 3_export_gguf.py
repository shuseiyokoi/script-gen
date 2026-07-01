#!/usr/bin/env python3
"""
Step 3: Merge LoRA adapter → full model → export to GGUF → register with Ollama.

Prerequisites:
  - llama.cpp installed (brew install llama.cpp  OR  build from source)
  - ollama installed

Steps performed:
  1. Merge LoRA adapter into base model weights
  2. Save merged model to output/merged/
  3. Convert to GGUF via llama.cpp convert script
  4. Quantize to Q4_K_M (optional, recommended)
  5. Create Modelfile and register with Ollama

Usage:
    python 3_export_gguf.py
    # then test with: ollama run podcast-scriptwriter
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2-2b-it")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "output/model")
MERGED_DIR = os.getenv("MERGED_DIR", "output/merged")
GGUF_DIR = os.getenv("GGUF_DIR", "output/gguf")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "podcast-scriptwriter")
QUANTIZE = os.getenv("QUANTIZE", "q4_k_m")  # set to "" to skip quantization
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Locate llama.cpp convert script
LLAMACPP_CONVERT = os.getenv(
    "LLAMACPP_CONVERT",
    shutil.which("convert_hf_to_gguf.py") or "/usr/local/lib/python3/dist-packages/llama_cpp/convert_hf_to_gguf.py",
)
LLAMACPP_QUANTIZE_BIN = os.getenv(
    "LLAMACPP_QUANTIZE",
    shutil.which("llama-quantize") or shutil.which("quantize") or "",
)


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def merge_adapter() -> None:
    print("\n[1/4] Merging LoRA adapter into base model …")
    if not Path(ADAPTER_DIR).exists():
        sys.exit(f"Adapter not found at {ADAPTER_DIR}. Run 2_finetune.py first.")

    Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN or None,
        device_map="cpu",
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, token=HF_TOKEN or None)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"  Merged model saved to {MERGED_DIR}")


def convert_to_gguf() -> Path:
    print("\n[2/4] Converting merged model to GGUF …")
    Path(GGUF_DIR).mkdir(parents=True, exist_ok=True)

    gguf_f16 = Path(GGUF_DIR) / "model-f16.gguf"

    if not Path(LLAMACPP_CONVERT).exists():
        sys.exit(
            f"llama.cpp convert script not found at {LLAMACPP_CONVERT}.\n"
            "Install llama.cpp:  brew install llama.cpp\n"
            "Or set LLAMACPP_CONVERT=/path/to/convert_hf_to_gguf.py"
        )

    run([sys.executable, LLAMACPP_CONVERT, MERGED_DIR, "--outfile", str(gguf_f16), "--outtype", "f16"])
    return gguf_f16


def quantize(gguf_f16: Path) -> Path:
    if not QUANTIZE:
        print("\n[3/4] Skipping quantization (QUANTIZE is empty).")
        return gguf_f16

    print(f"\n[3/4] Quantizing to {QUANTIZE.upper()} …")
    gguf_q = Path(GGUF_DIR) / f"model-{QUANTIZE}.gguf"

    if not LLAMACPP_QUANTIZE_BIN:
        print("  WARNING: llama-quantize binary not found. Skipping quantization.")
        print("  Install llama.cpp to enable quantization.")
        return gguf_f16

    run([LLAMACPP_QUANTIZE_BIN, str(gguf_f16), str(gguf_q), QUANTIZE.upper()])
    gguf_f16.unlink(missing_ok=True)  # remove large intermediate
    return gguf_q


def register_with_ollama(gguf_path: Path) -> None:
    print(f"\n[4/4] Registering {gguf_path.name} with Ollama as '{OLLAMA_MODEL_NAME}' …")

    modelfile_path = Path(GGUF_DIR) / "Modelfile"
    modelfile_content = f"""FROM {gguf_path.resolve()}

SYSTEM \"\"\"You are a professional podcast scriptwriter specializing in AI and technology.
Given a news article or summary, write an engaging podcast script segment (60-90 seconds when spoken).
The script should be conversational, informative, and accessible to a general tech-savvy audience.
Return ONLY the script text — no titles, no stage directions, no extra commentary.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
    modelfile_path.write_text(modelfile_content)

    if not shutil.which("ollama"):
        print(f"  Ollama not found in PATH. Modelfile written to {modelfile_path}")
        print(f"  Register manually:  ollama create {OLLAMA_MODEL_NAME} -f {modelfile_path}")
        return

    run(["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(modelfile_path)])
    print(f"\n  Model registered. Test it with:")
    print(f"    ollama run {OLLAMA_MODEL_NAME}")


def main() -> None:
    print("=== GGUF Export Pipeline ===")
    merge_adapter()
    gguf_f16 = convert_to_gguf()
    gguf_final = quantize(gguf_f16)
    register_with_ollama(gguf_final)
    print("\nAll done.")


if __name__ == "__main__":
    main()
