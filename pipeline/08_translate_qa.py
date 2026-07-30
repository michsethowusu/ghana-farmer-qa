#!/usr/bin/env python3
"""
Stage 8 -- Translate the English QA pairs into Ghanaian languages.

Two modes:
  direct  en -> target                (fewer hops, better fidelity)
  pivot   en -> pivot lang -> target  (routes through an intermediate language)

Every translation is validated as non-empty and different from its source, and
retried up to --max-retries times. Writes a snapshot of the full CSV plus a
checkpoint every --save-every batches, so an interrupted run resumes without
redoing finished work.

IMPORTANT -- language codes: Google Translate uses 'gaa' for Ga. Plain 'ga' is
Irish (Gaeilge) and returns valid-looking Irish text rather than erroring, so the
mistake is silent. See LANGS below; stage 10 checks for it.

Requirements: the py-googletrans fork -- pip install git+https://github.com/michsethowusu/py-googletrans
"""
import argparse
import asyncio
import json
import os
import sys
from typing import List

import pandas as pd
from googletrans import Translator

# Column suffix -> Google Translate language code.
LANGS = {"ee": "ee", "ga": "gaa", "ak": "ak"}


async def translate_validated(translator: Translator, texts: List[str], src: str, dest: str,
                              max_retries: int = 3) -> List[str]:
    """Translate texts, retrying any result that is empty or unchanged from its source."""
    if not texts:
        return []

    results = [""] * len(texts)
    for attempt in range(1, max_retries + 1):
        try:
            pending_idx, pending_txt = [], []
            for i, orig in enumerate(texts):
                cur = results[i]
                # A short source ("Yes", "50kg") can legitimately translate to itself,
                # so only treat identical output as a failure for longer strings.
                if not cur.strip() or (len(orig.strip()) >= 4 and cur.strip() == orig.strip()):
                    pending_idx.append(i)
                    pending_txt.append(orig)
            if not pending_idx:
                break

            out = await translator.translate(pending_txt, src=src, dest=dest)
            if not isinstance(out, list):
                out = [out]

            for i, trans in zip(pending_idx, out):
                text = trans.text.strip() if trans and trans.text else ""
                orig = texts[i]
                if text and (len(orig.strip()) < 4 or text != orig.strip()):
                    results[i] = text

        except Exception as e:
            if attempt == max_retries:
                print(f"Translation error ({src}->{dest}) after {max_retries} attempts: {e}",
                      file=sys.stderr)
            await asyncio.sleep(2 * attempt)

    return results


async def run(args):
    if args.fresh:
        for path in (args.output, args.checkpoint):
            if os.path.exists(path):
                os.remove(path)
                print(f"--fresh: removed {path}")

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.iloc[:args.limit].copy()
    total = len(df)
    print(f"Rows to translate: {total:,}")

    trans_cols = [f"{f}_{lang}" for lang in LANGS for f in ("question", "answer")]
    for col in trans_cols:
        df[col] = ""

    # Resume. The checkpoint and the CSV snapshot are written under the same lock,
    # so every batch in the checkpoint has its rows present in the CSV.
    done = set()
    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r", encoding="utf-8") as f:
                done = set(json.load(f).get("processed_batches", []))
        except Exception as e:
            print(f"Could not read checkpoint, starting over: {e}", file=sys.stderr)
            done = set()

        if done and os.path.exists(args.output):
            print(f"Resuming: {len(done)} batches done, reloading them from {args.output}...")
            prev = pd.read_csv(args.output, usecols=trans_cols, low_memory=False)
            if len(prev) == total:
                for col in trans_cols:
                    df[col] = prev[col].fillna("").astype(str)
            else:
                print(f"Row count mismatch ({len(prev)} vs {total}) -- ignoring old output.",
                      file=sys.stderr)
                done = set()
        elif done:
            print(f"Checkpoint found but {args.output} is missing -- starting over.", file=sys.stderr)
            done = set()

    batches = [(i // args.batch_size, list(range(i, min(i + args.batch_size, total))))
               for i in range(0, total, args.batch_size)]
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def process(batch_idx, indices):
        if batch_idx in done:
            return
        async with semaphore:
            questions = df.loc[indices, "question"].fillna("").astype(str).tolist()
            answers = df.loc[indices, "answer"].fillna("").astype(str).tolist()

            async with Translator() as t:
                if args.mode == "pivot":
                    q_mid = await translate_validated(t, questions, "en", args.pivot_lang, args.max_retries)
                    a_mid = await translate_validated(t, answers, "en", args.pivot_lang, args.max_retries)
                    src = args.pivot_lang
                else:
                    q_mid, a_mid, src = questions, answers, "en"

                q_out, a_out = {}, {}
                for lang, code in LANGS.items():
                    q_out[lang] = await translate_validated(t, q_mid, src, code, args.max_retries)
                    a_out[lang] = await translate_validated(t, a_mid, src, code, args.max_retries)

            async with lock:
                for lang in LANGS:
                    df.loc[indices, f"question_{lang}"] = q_out[lang]
                    df.loc[indices, f"answer_{lang}"] = a_out[lang]
                done.add(batch_idx)

                if batch_idx % args.save_every == 0:
                    df.to_csv(args.output, index=False)
                    with open(args.checkpoint, "w", encoding="utf-8") as cp:
                        json.dump({"processed_batches": list(done)}, cp)
                    print(f"Progress: batch {batch_idx}/{len(batches)} saved.", flush=True)

    route = (f"en -> {args.pivot_lang} -> {'/'.join(LANGS.values())}" if args.mode == "pivot"
             else f"en -> {'/'.join(LANGS.values())}")
    print(f"Starting {args.mode} translation ({route}) across {len(batches)} batches, "
          f"concurrency={args.concurrency}, batch_size={args.batch_size}...")
    await asyncio.gather(*[process(b, idx) for b, idx in batches])

    df.to_csv(args.output, index=False)
    with open(args.checkpoint, "w", encoding="utf-8") as cp:
        json.dump({"processed_batches": list(done)}, cp)
    print(f"Translation complete. Saved to {args.output}")
    print("Now run stage 10 to validate before publishing.")


def main():
    p = argparse.ArgumentParser(description="Translate QA pairs into Ghanaian languages")
    p.add_argument("--mode", choices=["direct", "pivot"], default="direct",
                   help="direct: en->target. pivot: en->pivot->target")
    p.add_argument("--pivot-lang", default="th", help="Intermediate language for --mode pivot")
    p.add_argument("--input", default="farmer_qa_flattened.csv", help="CSV from stage 7")
    p.add_argument("--output", default=None, help="Output CSV (default depends on --mode)")
    p.add_argument("--checkpoint", default=None, help="Checkpoint JSON (default depends on --mode)")
    p.add_argument("--batch-size", type=int, default=30, help="Rows per batch")
    p.add_argument("--concurrency", type=int, default=15, help="Concurrent batches")
    p.add_argument("--save-every", type=int, default=25, help="Snapshot the CSV every N batches")
    p.add_argument("--max-retries", type=int, default=3, help="Attempts per translation call")
    p.add_argument("--limit", type=int, default=None, help="Only translate N rows (testing)")
    p.add_argument("--fresh", action="store_true", help="Discard existing output and checkpoint")
    args = p.parse_args()

    if args.output is None:
        args.output = ("farmer_qa_pivot_translated.csv" if args.mode == "pivot"
                       else "farmer_qa_translated.csv")
    if args.checkpoint is None:
        args.checkpoint = ("translation_pivot_checkpoint.json" if args.mode == "pivot"
                           else "translation_checkpoint.json")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
