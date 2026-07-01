#!/usr/bin/env python3
"""
Step 1: Generate fine-tuning dataset.

Pulls AI/tech news articles from HuggingFace (JulesBelveze/tldr_news),
sends each article to a local Ollama model to generate a podcast script,
then saves the result as JSONL in chat format ready for fine-tuning.

Output: dataset/finetune.jsonl
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
OUTPUT_PATH = Path("dataset/finetune.jsonl")
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "500"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

SYSTEM_PROMPT = """You are a professional podcast scriptwriter specializing in AI and technology.
Given a news article or summary, write an engaging podcast script segment (60-90 seconds when spoken).
The script should be conversational, informative, and accessible to a general tech-savvy audience.
Return ONLY the script text — no titles, no stage directions, no extra commentary."""

USER_TEMPLATE = """Write a podcast script segment for this article:

Title: {title}
Category: {category}
Summary: {content}"""


def generate_script(title: str, category: str, content: str) -> str | None:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                title=title,
                category=category,
                content=content[:2000],
            )},
        ],
        "stream": False,
        "options": {"temperature": 0.7},
    }
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"  [error] Ollama call failed: {e}", file=sys.stderr)
        return None


def main() -> None:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print(f"Loading dataset JulesBelveze/tldr_news ...")
    ds = load_dataset("JulesBelveze/tldr_news", split="train", trust_remote_code=True)
    print(f"  {len(ds)} total records")

    # Filter to AI category if available
    if "category" in ds.column_names:
        ai_ds = ds.filter(lambda x: (x.get("category") or "").lower() in ("ai", "tech", "science"))
        print(f"  {len(ai_ds)} after AI/tech/science filter")
    else:
        ai_ds = ds

    records = list(ai_ds)[:MAX_ARTICLES]
    print(f"  Using {len(records)} articles (MAX_ARTICLES={MAX_ARTICLES})")

    # Count existing lines so we can resume
    existing = 0
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        print(f"  Resuming from {existing} existing examples")

    written = 0
    skipped = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for i, row in enumerate(tqdm(records, desc="Generating scripts")):
            if i < existing:
                continue

            title = (row.get("headline") or row.get("title") or "").strip()
            content = (row.get("summary") or row.get("content") or row.get("text") or "").strip()
            category = (row.get("category") or "AI").strip()

            if not title or not content:
                skipped += 1
                continue

            script = generate_script(title, category, content)
            if not script:
                skipped += 1
                continue

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        title=title,
                        category=category,
                        content=content[:2000],
                    )},
                    {"role": "assistant", "content": script},
                ]
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            out.flush()
            written += 1

            # Small pause to avoid overloading Ollama
            time.sleep(0.1)

    print(f"\nDone. Written {written} new examples, skipped {skipped}.")
    print(f"Dataset saved to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
