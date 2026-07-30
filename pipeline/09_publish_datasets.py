#!/usr/bin/env python3
"""
Stage 9 -- Split the translated CSV into one dataset per language and publish them.

For each target language, writes a parquet holding the English columns plus that
language's two translated columns, generates a dataset card, and pushes to one or
more Hugging Face orgs. Cards are generated per org so sibling links stay in-org.

  --build-only   write the parquet files and cards locally, push nothing
  --cards-only   re-push just the READMEs (leaves parquet files untouched)

Authentication: cached Hugging Face login (`hf auth login`) or HF_TOKEN.
"""
import argparse
import os
import sys

import pandas as pd
from huggingface_hub import HfApi

LANGS = {
    "ewe": {"suffix": "ee", "code": "ee",  "name": "Ewe"},
    "ga":  {"suffix": "ga", "code": "gaa", "name": "Ga"},
    "twi": {"suffix": "ak", "code": "ak",  "name": "Twi (Akan)"},
}
ENGLISH_COLS = ["question", "answer", "category", "source_text", "original_twi", "file_name"]

PROVENANCE = """## Where the content comes from

1. **Field recordings.** Ghanaian farmers were recorded discussing their practice
   in Twi -- cocoa spacing, pest pressure, soil fertility, harvest and pricing.
   These were transcribed and translated into English, published as
   [`{source_repo}`](https://huggingface.co/datasets/{source_repo}).
2. **Question generation.** For each of the {n_src:,} interview passages, a model
   was asked to write ~10 standalone questions a farmer might put to an extension
   officer about the practices described in *that* passage, plus an explanatory
   answer for each. Each question had to stand on its own -- no follow-ups, no
   references to "the text above" -- so the pairs are usable out of context.
3. **Flattening.** One row per Q&A pair, carrying its source passage forward."""

CARD = """---
license: {license}
language:
- en
- {code}
task_categories:
- question-answering
- translation
source_datasets:
- {org}/{en_repo}
tags:
- agriculture
- ghana
- african-languages
- low-resource
- parallel-corpus
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Ghana Farmer Q&A -- English / {name}

**{n:,} parallel English-{name} question-answer pairs on Ghanaian smallholder
farming, grounded in {n_src:,} recorded interviews with Ghanaian farmers.**

This is the {name} edition of
[`{org}/{en_repo}`](https://huggingface.co/datasets/{org}/{en_repo}): the same
farmer-grounded Q&A pairs, with each question and answer aligned to a {name}
translation. Every row holds the English and the {name} side by side, plus the
interview passage the pair came from.

{provenance}
4. **Translation.** The English question and answer were machine-translated into
   {name} (Google Translate, `{code}`){route}, with each output checked as
   non-empty and distinct from its source and retried on failure. Rows with
   untranslated residue were dropped.

## Columns

| column | description |
|---|---|
| `question` | Farming question in the first person, English |
| `answer` | Explanatory answer, English |
| `category` | Topic label chosen by the model, e.g. "Pest & Disease Control" |
| `source_text` | The interview passage this pair was generated from, in English |
| `original_twi` | The original Twi transcription of that passage |
| `file_name` | Source audio file identifier |
| `question_{suffix}` | `question` in {name} |
| `answer_{suffix}` | `answer` in {name} |

## What it's useful for

{name} is a low-resource language with little parallel text available, and almost
none in a practical domain like agriculture. That makes this useful for:

- **Training and evaluating English<->{name} translation**, especially in-domain
  agricultural and advisory language that general corpora don't cover.
- **Building {name}-language farmer advisory tools** -- chat assistants, IVR and
  voice systems, extension material -- where the question register matters as
  much as the content.
- **Multilingual QA**: the same questions exist in English and each target
  language, so you can compare model behaviour across languages on identical
  content.
- **Terminology mining** for agricultural vocabulary in {name} -- crop names,
  pests, practices -- which is thin in existing resources.
- **A post-editing base.** Machine translation gets you coverage cheaply; human
  review is the fastest route to a gold-standard {name} corpus, and the English
  source sits in the same row to make review straightforward.

## Limitations

- **The {name} text is unreviewed machine translation.** Google Translate support
  for {name} is recent and uneven, so expect errors and treat the {name} side as a
  starting point rather than a reference.{pivot_note}
- **The English answers are model-written and not expert-reviewed.** They are
  grounded in farmer interviews, but no agronomist verified them. Do not deploy
  as extension advice without review.
- **`category` is free text**, with heavy near-duplication. Cluster or remap it
  before using it as a label.
- **Interview coverage shapes topic coverage** -- this is not a balanced survey of
  Ghanaian agriculture.

## License

{license_note}
"""


def main():
    p = argparse.ArgumentParser(description="Split translated QA into per-language datasets and publish")
    p.add_argument("--input", default="farmer_qa_pivot_translated.csv", help="Validated CSV from stage 8/10")
    p.add_argument("--orgs", default="", help="Comma-separated HF orgs to publish to")
    p.add_argument("--repo-prefix", default="ghana-farmer-qa", help="Repo name prefix")
    p.add_argument("--en-repo", default="ghana-farmer-qa", help="Repo name of the English original")
    p.add_argument("--source-repo", default="ghananlpcommunity/twi-english-agric",
                   help="Upstream Twi-English corpus, for the provenance section")
    p.add_argument("--build-dir", default="hf_build", help="Where to write parquet files and cards")
    p.add_argument("--license", default="cc-by-nc-4.0", help="License identifier for the cards")
    p.add_argument("--pivot-lang", default=None,
                   help="If the input came from pivot mode, name the pivot language for the card")
    p.add_argument("--private", action="store_true", help="Create repos private")
    p.add_argument("--build-only", action="store_true", help="Build locally, push nothing")
    p.add_argument("--cards-only", action="store_true", help="Only re-push READMEs")
    args = p.parse_args()

    orgs = [o.strip() for o in args.orgs.split(",") if o.strip()]
    if not orgs and not args.build_only:
        sys.exit("Specify --orgs, or use --build-only.")

    api = None
    if not args.build_only:
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        try:
            me = api.whoami()
        except Exception:
            sys.exit("Not authenticated with Hugging Face. Run `hf auth login` or set HF_TOKEN.")
        member = {o["name"] for o in me.get("orgs", [])}
        for org in orgs:
            if org != me["name"] and org not in member:
                sys.exit(f"Account {me['name']} is not a member of '{org}'.")

    df = pd.read_csv(args.input, low_memory=False)
    n_src = df["file_name"].nunique()
    print(f"{args.input}: {len(df):,} rows, {n_src:,} source passages")

    route = f" via a {args.pivot_lang} pivot" if args.pivot_lang else ""
    pivot_note = (f" English was translated via a {args.pivot_lang} pivot, which adds its own "
                  f"loss of fidelity." if args.pivot_lang else "")
    license_note = (f"{args.license.upper()}, inherited from the source dataset "
                    f"[`{args.source_repo}`](https://huggingface.co/datasets/{args.source_repo}).")

    for slug, meta in LANGS.items():
        s = meta["suffix"]
        cols = ENGLISH_COLS + [f"question_{s}", f"answer_{s}"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            sys.exit(f"{slug}: missing columns {missing}")

        d = os.path.join(args.build_dir, slug, "data")
        os.makedirs(d, exist_ok=True)
        pq = os.path.join(d, "train-00000-of-00001.parquet")
        if not args.cards_only:
            df[cols].to_parquet(pq, index=False, compression="snappy")
            print(f"{slug:4s} rows={len(df):,} parquet={os.path.getsize(pq) / 1e6:.1f} MB")

        for org in (orgs or ["_local"]):
            card = CARD.format(
                license=args.license, code=meta["code"], name=meta["name"], suffix=s,
                org=org, en_repo=args.en_repo, n=len(df), n_src=n_src, route=route,
                pivot_note=pivot_note, license_note=license_note,
                provenance=PROVENANCE.format(source_repo=args.source_repo, n_src=n_src))
            card_path = os.path.join(args.build_dir, slug, f"README.{org}.md")
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(card)

            if args.build_only:
                continue

            repo_id = f"{org}/{args.repo_prefix}-{slug}"
            api.create_repo(repo_id, repo_type="dataset", private=args.private, exist_ok=True)
            if args.cards_only:
                api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                                repo_id=repo_id, repo_type="dataset",
                                commit_message="Update dataset card")
            else:
                api.upload_file(path_or_fileobj=pq,
                                path_in_repo="data/train-00000-of-00001.parquet",
                                repo_id=repo_id, repo_type="dataset",
                                commit_message=f"Add {meta['name']} parallel QA dataset")
                api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                                repo_id=repo_id, repo_type="dataset",
                                commit_message="Add dataset card")
            print(f"  pushed https://huggingface.co/datasets/{repo_id}")

    if args.build_only:
        print(f"\nBuilt locally in {args.build_dir}/ -- nothing pushed.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
