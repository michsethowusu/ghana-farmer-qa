#!/usr/bin/env python3
"""
Stage 3 -- Translate Twi transcriptions into English with Gemini.

Reads the transcription CSV from stage 2 and appends
file_name,original_twi,translated_english rows to the output CSV.

Resumes safely: any file_name already in the output CSV is skipped.

Needs GEMINI_API_KEY in the environment.
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm as async_tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

FAILED_MARKER = "[TRANSLATION FAILED]"


def build_prompt(twi_text):
    return f"""Translate the following transcribed Twi text from agriculture domain into English.
Output only the English translation, with no extra commentary or text.

Twi text: "{twi_text}"
"""


async def translate_one(client, model, config, file_name, twi_text, sem, max_retries, backoff):
    async with sem:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=build_prompt(twi_text))])]
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=model, contents=contents, config=config)
                return file_name, twi_text, response.text.strip()
            except Exception as e:
                logger.warning(f"{file_name}: attempt {attempt} failed ({e})")
                if attempt == max_retries:
                    return file_name, twi_text, None
                await asyncio.sleep(backoff * attempt)
    return file_name, twi_text, None


def append_rows(path, rows):
    df = pd.DataFrame(rows, columns=["file_name", "original_twi", "translated_english"])
    df.to_csv(path, mode="a", header=not Path(path).exists(), index=False)


def load_done(path):
    """file_names already translated, excluding ones marked as failed so they retry."""
    if not Path(path).exists():
        return set()
    try:
        df = pd.read_csv(path)
        if "file_name" not in df.columns:
            return set()
        ok = df[df["translated_english"].astype(str) != FAILED_MARKER]
        return set(ok["file_name"].unique())
    except Exception as e:
        logger.error(f"Could not read {path}: {e}. Starting fresh.")
        return set()


async def run(args):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. export it before running this stage.")

    df = pd.read_csv(args.input)
    for col in (args.file_col, args.text_col):
        if col not in df.columns:
            sys.exit(f"Missing column '{col}' in {args.input}. Found: {list(df.columns)}")
    logger.info(f"{len(df)} entries in {args.input}")

    done = load_done(args.output)
    logger.info(f"Already translated: {len(done)}")

    pending = [(r[args.file_col], r[args.text_col]) for _, r in df.iterrows()
               if r[args.file_col] not in done]
    if args.limit:
        pending = pending[:args.limit]
    logger.info(f"To translate this run: {len(pending)}")
    if not pending:
        return

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [asyncio.create_task(
        translate_one(client, args.model, config, fname, twi, sem, args.max_retries, args.backoff))
        for fname, twi in pending]

    buffer = []
    pbar = async_tqdm(total=len(tasks), desc="Translating")
    for coro in asyncio.as_completed(tasks):
        fname, twi, trans = await coro
        buffer.append({"file_name": fname, "original_twi": twi,
                       "translated_english": trans if trans else FAILED_MARKER})
        pbar.update(1)
        if len(buffer) >= args.batch_size:
            append_rows(args.output, buffer)
            buffer.clear()
    if buffer:
        append_rows(args.output, buffer)
    pbar.close()
    logger.info(f"Done. English translations in {args.output}")


def main():
    p = argparse.ArgumentParser(description="Translate Twi transcriptions to English")
    p.add_argument("--input", default="metadata.csv", help="CSV from stage 2")
    p.add_argument("--output", default="english_translations.csv", help="Output CSV")
    p.add_argument("--file-col", default="file", help="Identifier column in the input")
    p.add_argument("--text-col", default="transcription", help="Twi text column in the input")
    p.add_argument("--model", default="gemini-3.1-flash-lite", help="Gemini model")
    p.add_argument("--concurrency", type=int, default=5, help="Concurrent requests")
    p.add_argument("--batch-size", type=int, default=20, help="Rows buffered before each CSV append")
    p.add_argument("--max-retries", type=int, default=3, help="Attempts per row")
    p.add_argument("--backoff", type=float, default=2.0, help="Retry backoff seconds (multiplied by attempt)")
    p.add_argument("--limit", type=int, default=None, help="Only translate N rows (testing)")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
