#!/usr/bin/env python3
"""
Stage 10 -- Validate the translated QA CSV before publishing.

Checks each translation column for the failure modes this pipeline has actually
hit in practice:

  empty            cell is blank
  same-as-source   translation identical to the English input (nothing happened)
  pivot residue    pivot-language script left in the output, whole cell or partial
  wrong language   e.g. Google Translate 'ga' returns Irish, not Ga -- catching
                   this needs a positive check for the target's own orthography

Exits non-zero if any check fails, so run_pipeline.sh stops before publishing.
Use --write-clean to drop offending rows and write a cleaned CSV.
"""
import argparse
import re
import sys

import pandas as pd

# Script blocks that should never survive translation into a Ghanaian language.
PIVOT_SCRIPTS = {
    "Thai": r"[฀-๿]",
    "CJK": r"[一-鿿]",
    "Cyrillic": r"[Ѐ-ӿ]",
    "Arabic": r"[؀-ۿ]",
    "Devanagari": r"[ऀ-ॿ]",
}

# Irish function words -- the tell-tale of the 'ga' (Irish) vs 'gaa' (Ga) mix-up.
IRISH = r"\b(?:cén|bhfuil|chrann|agus|faoi|nach|ina|maidin|mhaith)\b"

# Ghanaian orthography: Ewe, Ga and Twi all use these extended Latin letters.
GHANAIAN_GLYPHS = r"[ɛɔŋƒɖʋã]"


def main():
    p = argparse.ArgumentParser(description="Validate translated QA output")
    p.add_argument("--input", default="farmer_qa_pivot_translated.csv", help="CSV to validate")
    p.add_argument("--langs", default="ee,ga,ak", help="Comma-separated column suffixes")
    p.add_argument("--min-glyph-fraction", type=float, default=0.90,
                   help="Fail if fewer than this fraction of cells carry Ghanaian glyphs")
    p.add_argument("--write-clean", default=None, help="Write a cleaned CSV with bad rows dropped")
    args = p.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    cols = [f"{field}_{lang}" for lang in langs for field in ("question", "answer")]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"Missing expected columns: {missing}")

    print(f"{args.input}: {len(df):,} rows, checking {len(cols)} translation columns\n")

    failures = []
    bad_row_mask = pd.Series(False, index=df.index)

    for col in cols:
        src = "question" if col.startswith("question") else "answer"
        s = df[col].astype(str)
        issues = []

        empty = s.str.strip().eq("") | df[col].isna()
        if empty.any():
            issues.append(f"{empty.sum():,} empty")
            bad_row_mask |= empty

        same = s.str.strip() == df[src].astype(str).str.strip()
        if same.any():
            issues.append(f"{same.sum():,} identical to source")
            bad_row_mask |= same

        for name, pattern in PIVOT_SCRIPTS.items():
            hit = s.str.contains(pattern, regex=True, na=False)
            if hit.any():
                issues.append(f"{hit.sum():,} with {name} residue")
                bad_row_mask |= hit

        irish = s.str.contains(IRISH, regex=True, case=False, na=False)
        if irish.any():
            issues.append(f"{irish.sum():,} look like Irish (is the code 'ga' instead of 'gaa'?)")
            bad_row_mask |= irish

        glyph_frac = s.str.contains(GHANAIAN_GLYPHS, regex=True, na=False).mean()

        status = "FAIL" if issues else "ok"
        if issues:
            failures.append((col, issues))
        print(f"  {status:4s} {col:14s} glyph coverage {glyph_frac:5.1%}  {'; '.join(issues)}")

        if glyph_frac < args.min_glyph_fraction:
            msg = (f"only {glyph_frac:.1%} of cells contain Ghanaian orthography "
                   f"(expected >= {args.min_glyph_fraction:.0%}) -- possibly the wrong language")
            failures.append((col, [msg]))
            print(f"  FAIL {col:14s} {msg}")

    print()
    if args.write_clean:
        clean = df[~bad_row_mask]
        clean.to_csv(args.write_clean, index=False)
        print(f"Dropped {int(bad_row_mask.sum()):,} row(s); wrote {len(clean):,} to {args.write_clean}")

    if failures:
        print(f"VALIDATION FAILED -- {len(failures)} column check(s) with issues.")
        if not args.write_clean:
            print("Re-run with --write-clean <path> to drop the offending rows.")
        sys.exit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()
