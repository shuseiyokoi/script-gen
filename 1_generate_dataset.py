#!/usr/bin/env python3
"""
Step 1: Generate fine-tuning dataset.

Pulls AI/tech news articles from HuggingFace (JulesBelveze/tldr_news),
sends each article to a local Ollama model to generate a two-host podcast
conversation (expert + learner), 5-10 minutes long depending on article
length, then saves the result as JSONL in chat format ready for fine-tuning.

Output: dataset/finetune_dialogue.jsonl
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
OUTPUT_PATH = Path("dataset/finetune_dialogue.jsonl")
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "500"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# Two recurring hosts, matching a two-person "explainer" podcast format:
# an expert who knows the topic professionally, and a learner who asks
# the questions a listener would ask.
EXPERT_NAME = "Dr. Lena Osei"
LEARNER_NAME = "Kai"

WORDS_PER_MINUTE = 150  # conversational speaking pace, combined across both speakers
MIN_MINUTES, MAX_MINUTES = 5, 10
# Content word counts below this map to MIN_MINUTES, at/above this map to MAX_MINUTES.
SHORT_ARTICLE_WORDS, LONG_ARTICLE_WORDS = 50, 400

SYSTEM_PROMPT = f"""You are writing scripts for a two-host AI podcast in the style of an AI-generated \
"deep dive" explainer show. The two recurring hosts are:

- {EXPERT_NAME}: a professional subject-matter expert on the topic. Explains clearly, gives context, \
corrects misconceptions, and shares real-world implications.
- {LEARNER_NAME}: a curious, intelligent non-expert who is learning about the topic live. Asks the \
questions a listener would ask, pushes for clarification, reacts, and occasionally summarizes what \
they just learned in their own words.

Given a news article, write a natural back-and-forth conversation between the two hosts that explains \
the article in depth. The conversation must run for the target length and word count given in the \
request, scaled to how much substance is in the source article — denser articles get closer to the \
10-minute end, shorter ones closer to 5 minutes.

Formatting rules:
- Each line starts with the speaker's name followed by a colon, e.g. "{EXPERT_NAME}: ..." or "{LEARNER_NAME}: ...".
- Alternate naturally based on conversation flow — do not force strict turn-taking.
- No episode titles, no stage directions, no sound effect cues, no music cues, no extra commentary outside the dialogue.
- Open with {LEARNER_NAME} introducing the topic or asking what it's about, and close with a brief \
takeaway line from {EXPERT_NAME}."""

USER_TEMPLATE = """Write a podcast conversation for this article.

Title: {title}
Category: {category}
Summary: {content}

Target length: approximately {target_minutes} minutes when spoken aloud (~{target_words} words total \
across both speakers)."""


def target_length(content: str) -> tuple[int, int]:
    """Scale target podcast duration (minutes, words) to the length of the source article."""
    word_count = len(content.split())
    span = max(LONG_ARTICLE_WORDS - SHORT_ARTICLE_WORDS, 1)
    frac = (word_count - SHORT_ARTICLE_WORDS) / span
    frac = min(max(frac, 0.0), 1.0)
    minutes = MIN_MINUTES + frac * (MAX_MINUTES - MIN_MINUTES)
    minutes = round(minutes)
    words = minutes * WORDS_PER_MINUTE
    return minutes, words


def generate_script(title: str, category: str, content: str, target_minutes: int, target_words: int) -> str | None:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                title=title,
                category=category,
                content=content[:2000],
                target_minutes=target_minutes,
                target_words=target_words,
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

            target_minutes, target_words = target_length(content)
            script = generate_script(title, category, content, target_minutes, target_words)
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
                        target_minutes=target_minutes,
                        target_words=target_words,
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
