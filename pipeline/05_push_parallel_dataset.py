#!/usr/bin/env python3
"""
Stage 5 (optional) -- Publish the Twi-English parallel corpus to the Hub.

Uploads the stage-3 output as a text dataset with a dataset card. This is the
corpus the QA stages read from, so publishing it makes the QA datasets traceable.

Authentication: cached Hugging Face login (`hf auth login`) or HF_TOKEN.
"""
import argparse
import os
import sys
import tempfile

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

CARD = """---
language:
- tw
- en
license: cc-by-nc-4.0
task_categories:
- translation
pretty_name: Twi-English Parallel Corpus for Agriculture
size_categories:
- 10K<n<100K
---

# Twi-English Parallel Corpus for Agriculture

A parallel corpus of **Twi (Akan)** transcriptions and their **English
translations**, in the agricultural domain. Audio was extracted from YouTube
videos, split into {chunk_sec}-second chunks, transcribed, and machine-translated.

- **Domain:** agriculture -- cocoa farming, plantain cultivation, general practice
- **Source:** YouTube video audio
- **Filename convention:** `<YouTubeVideoID>_<segment>.wav`, e.g.
  `P74kzqm2nno_035.wav` comes from video `P74kzqm2nno`, so every row is traceable
  back to its source video and offset.

## Fields

- `file_name` -- audio segment filename (contains the video ID and segment number)
- `original_twi` -- Twi transcription of the spoken audio
- `translated_english` -- English translation of that transcription

## Limitations

- Transcriptions come from an automatic speech recognizer on spontaneous speech,
  so expect recognition errors; chunks that failed recognition are absent.
- Translations are machine-generated and unreviewed.
- 30-second chunking can cut sentences mid-utterance, so a row may start or end
  mid-thought.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0])
```

## License

CC BY-NC 4.0.
"""


def main():
    p = argparse.ArgumentParser(description="Push the Twi-English parallel corpus to the Hub")
    p.add_argument("--repo-id", required=True, help="Target repo, e.g. myorg/twi-english-agric")
    p.add_argument("--input", default="english_translations.csv", help="CSV from stage 3")
    p.add_argument("--chunk-sec", type=int, default=30, help="Chunk length used in stage 1 (for the card)")
    p.add_argument("--private", action="store_true", help="Create the repo private")
    p.add_argument("--drop-failed", action="store_true",
                   help="Drop rows whose translation is [TRANSLATION FAILED]")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception:
        sys.exit("Not authenticated with Hugging Face. Run `hf auth login` or set HF_TOKEN.")

    df = pd.read_csv(args.input)
    if args.drop_failed:
        before = len(df)
        df = df[df["translated_english"].astype(str) != "[TRANSLATION FAILED]"]
        print(f"Dropped {before - len(df)} failed row(s).")
    print(f"Pushing {len(df)} rows to {args.repo_id}")

    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    DatasetDict({"train": Dataset.from_pandas(df, preserve_index=False)}).push_to_hub(
        args.repo_id, private=args.private, token=token,
        commit_message="Upload Twi-English agriculture parallel corpus")

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(CARD.format(repo_id=args.repo_id, chunk_sec=args.chunk_sec))
        card_path = f.name
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="dataset", token=token,
                    commit_message="Add dataset card")
    os.unlink(card_path)

    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
