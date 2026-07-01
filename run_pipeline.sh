#!/bin/bash
# Runs step 2 (fine-tune) and step 3 (GGUF export) back to back.
# Exits non-zero on first failure. Assumes dataset/finetune.jsonl already exists.
set -e
cd "C:\Users\myoub\projects\script-gen"

LINES=$(wc -l < dataset/finetune.jsonl)
echo "Dataset has $LINES lines."
if [ "$LINES" -lt 50 ]; then
    echo "FATAL: dataset only has $LINES lines — aborting pipeline."
    exit 1
fi

echo "=== Step 2: fine-tuning ==="
.venv/Scripts/python.exe 2_finetune.py

echo "=== Step 3: GGUF export ==="
export LLAMACPP_CONVERT="$(pwd)/.llama.cpp/convert_hf_to_gguf.py"
export LLAMACPP_QUANTIZE="$(pwd)/.llama.cpp-bin/llama-quantize.exe"
.venv/Scripts/python.exe 3_export_gguf.py

echo "=== Pipeline complete ==="
