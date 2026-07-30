#!/usr/bin/env python3
"""
Stage 4 (optional) -- Publish the Twi speech dataset (audio + transcription) to the Hub.

Uploads the WAV chunks alongside their Twi transcriptions as an audio dataset.
Optional: the QA pipeline does not consume this, but it publishes the ASR corpus
the rest of the work is built on.

Authentication: uses your cached Hugging Face login (`hf auth login`) or the
HF_TOKEN environment variable. Never hardcode a token in this file.

Requirements: datasets, huggingface_hub, soundfile, librosa.
"""
import argparse
import csv
import os
import sys

from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import HfApi


def load_metadata(csv_path, audio_dir):
    records, missing = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = os.path.join(audio_dir, row["file"])
            if not os.path.exists(path):
                missing.append(row["file"])
                continue
            records.append({"file": path, "transcription": row["transcription"].strip()})
    if missing:
        print(f"{len(missing)} file(s) listed in the CSV are not on disk -- skipped.")
        for m in missing[:10]:
            print(f"   - {m}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")
    print(f"Loaded {len(records)} record(s).")
    return records


def main():
    p = argparse.ArgumentParser(description="Push the Twi speech dataset to the Hub")
    p.add_argument("--repo-id", required=True, help="Target dataset repo, e.g. myorg/twi-agriculture-asr")
    p.add_argument("--metadata", default="metadata.csv", help="CSV from stage 2")
    p.add_argument("--audio-dir", default="audio_chunks", help="Directory holding the WAV chunks")
    p.add_argument("--split", default="train", help="Split name")
    p.add_argument("--sampling-rate", type=int, default=16000, help="Target sampling rate")
    p.add_argument("--private", action="store_true", help="Create the repo private")
    p.add_argument("--card", default=None, help="Optional path to a README.md to upload as the card")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or None  # None => fall back to cached login
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception:
        sys.exit("Not authenticated with Hugging Face. Run `hf auth login` or set HF_TOKEN.")

    records = load_metadata(args.metadata, args.audio_dir)
    if not records:
        sys.exit("No valid records. Check --metadata and --audio-dir.")

    ds = Dataset.from_dict({
        "audio": [r["file"] for r in records],
        "transcription": [r["transcription"] for r in records],
    }).cast_column("audio", Audio(sampling_rate=args.sampling_rate))
    dsd = DatasetDict({args.split: ds})
    print(dsd)

    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")

    print("Uploading (audio uploads can take a while)...")
    dsd.push_to_hub(args.repo_id, token=token,
                    commit_message="Add Twi agriculture speech dataset")

    if args.card and os.path.exists(args.card):
        api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="dataset", token=token,
                        commit_message="Add dataset card")
        print("Card uploaded.")

    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
