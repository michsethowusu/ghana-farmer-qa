#!/usr/bin/env python3
"""
Stage 7 -- Flatten the generated QA JSONL into one row per question-answer pair.

Stage 6 writes one JSONL record per source passage, each holding several category
groups which each hold several QA pairs. This explodes that nesting into a flat
CSV -- one row per QA pair -- carrying the source passage forward so every pair
stays traceable to the interview text it came from.

  1 JSONL record  ->  N categories  ->  M pairs each  ->  N*M CSV rows
"""
import argparse
import json
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Flatten generated QA JSONL into a CSV")
    p.add_argument("--input", default="farmer_qa_output.jsonl", help="JSONL from stage 6")
    p.add_argument("--output", default="farmer_qa_flattened.csv", help="Flat CSV output")
    args = p.parse_args()

    rows = []
    skipped = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {line_no}: malformed JSON, skipped ({e})", file=sys.stderr)
                skipped += 1
                continue

            source_text = rec.get("translated_english", "")
            original_twi = rec.get("original_twi", "")
            file_name = rec.get("file_name", "")

            for cat in rec.get("qa_data", []):
                category = cat.get("category", "")
                for qa in cat.get("qa_pairs", []):
                    question = (qa.get("question") or "").strip()
                    # stage 6 names the answer field "knowledge"
                    answer = (qa.get("knowledge") or "").strip()
                    if not question or not answer:
                        skipped += 1
                        continue
                    rows.append({
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "source_text": source_text,
                        "original_twi": original_twi,
                        "file_name": file_name,
                    })

    if not rows:
        sys.exit(f"No QA pairs found in {args.input}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"QA pairs written : {len(df):,}")
    print(f"source passages  : {df['file_name'].nunique():,}")
    print(f"skipped (empty/malformed): {skipped:,}")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
