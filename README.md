# script-gen — Podcast Script Fine-Tuning Toolkit

Three scripts that take you from zero to a custom Ollama model for podcast script generation.

```
1_generate_dataset.py   Pull TLDR news → generate scripts via Ollama → save JSONL
2_finetune.py           Fine-tune a small LLM on the dataset (QLoRA, single GPU)
3_export_gguf.py        Merge adapter → convert to GGUF → register with Ollama
```

---

## Quick Start

```bash
# 1. Create a virtual env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate the dataset (needs Ollama + gemma3:12b running)
python 1_generate_dataset.py

# 3. Fine-tune (needs a CUDA GPU; CPU works but is slow)
python 2_finetune.py

# 4. Export to Ollama
python 3_export_gguf.py
# → ollama run podcast-scriptwriter
```

---

## Environment Variables

### Step 1 — Dataset generation

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `gemma3:12b` | Model to use for script generation |
| `MAX_ARTICLES` | `500` | Number of articles to process |
| `OLLAMA_TIMEOUT` | `120` | Seconds per request |

### Step 2 — Fine-tuning

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL` | `google/gemma-2-2b-it` | HuggingFace model to fine-tune |
| `DATASET_PATH` | `dataset/finetune.jsonl` | Path to JSONL dataset |
| `OUTPUT_DIR` | `output/model` | Where to save LoRA adapter |
| `NUM_EPOCHS` | `3` | Training epochs |
| `BATCH_SIZE` | `2` | Per-device batch size |
| `HF_TOKEN` | `` | HuggingFace token (for gated models) |

### Step 3 — Export

| Variable | Default | Description |
|---|---|---|
| `ADAPTER_DIR` | `output/model` | LoRA adapter from step 2 |
| `OLLAMA_MODEL_NAME` | `podcast-scriptwriter` | Name in Ollama registry |
| `QUANTIZE` | `q4_k_m` | GGUF quantization level (set to `""` to skip) |
| `LLAMACPP_CONVERT` | auto-detected | Path to `convert_hf_to_gguf.py` |

---

## Dataset Format (JSONL)

Each line is a JSON object in OpenAI chat format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a professional podcast scriptwriter..."},
    {"role": "user",   "content": "Write a podcast script segment for this article:\n\nTitle: ..."},
    {"role": "assistant", "content": "The generated script text..."}
  ]
}
```

---

## Hardware Requirements

| Step | Minimum | Recommended |
|---|---|---|
| Generate dataset | CPU + 8 GB RAM | — |
| Fine-tune | 16 GB VRAM (RTX 3090) | 24 GB VRAM (RTX 4090) |
| Export/quantize | CPU + 16 GB RAM | — |

For CPU-only fine-tuning set `BATCH_SIZE=1` and expect ~1 hr/epoch on gemma-2-2b-it.

---

## Directory Structure (after running all steps)

```
script-gen/
├── dataset/
│   └── finetune.jsonl        # generated training data
├── output/
│   ├── model/                # LoRA adapter weights
│   ├── merged/               # merged full model
│   └── gguf/
│       ├── model-q4_k_m.gguf # quantized model
│       └── Modelfile         # Ollama Modelfile
├── 1_generate_dataset.py
├── 2_finetune.py
├── 3_export_gguf.py
├── requirements.txt
└── README.md
```
