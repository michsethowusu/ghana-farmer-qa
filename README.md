# Ghana Farmer QA

A pipeline that turns **YouTube videos of Ghanaian farmers talking about their
work** into **published parallel question-answer datasets** in English, Ewe, Ga
and Twi.

The point is grounding: every QA pair traces back to something a farmer actually
said on camera, and the source passage travels with it through every stage, so a
published answer can always be checked against the speech it came from.

```
YouTube URLs
   │  01  download audio, split into 30s WAV chunks
   ▼
audio_chunks/*.wav
   │  02  speech recognition (Twi)
   ▼
metadata.csv                     file, transcription
   │  03  Gemini translation
   ▼
english_translations.csv         file_name, original_twi, translated_english
   │        └── 04, 05  (optional) publish the speech + parallel corpora
   │  06  Gemini QA generation, ~10 standalone pairs per passage
   ▼
farmer_qa_output.jsonl           nested: passage → categories → QA pairs
   │  07  flatten
   ▼
farmer_qa_flattened.csv          one row per QA pair, + source passage
   │  08  translate into Ewe / Ga / Twi
   ▼
farmer_qa_translated.csv
   │  10  validate  ── fails the run if translations are empty,
   │                   unchanged, wrong-language, or hold pivot residue
   ▼
farmer_qa_clean.csv
   │  09  split per language, write cards, push
   ▼
Hugging Face datasets
```

## Quick start

```bash
pip install -r requirements.txt          # also needs ffmpeg on PATH
export GEMINI_API_KEY=...                # stages 3 and 6
hf auth login                            # stages 4, 5, 9

./run_pipeline.sh                        # every stage
./run_pipeline.sh 6 10                   # only stages 6-10
ORGS=myorg ./run_pipeline.sh 9 9         # publish only
```

Every stage checkpoints and resumes, so re-running after an interrupt picks up
where it stopped rather than redoing work. Each script also runs standalone with
`--help`.

## Stages

| # | Script | Does | Key inputs → output |
|---|---|---|---|
| 1 | `01_download_youtube_audio.py` | Downloads audio, converts to 16 kHz mono, slices into fixed chunks named `<videoID>_<NNN>.wav` | `urls.txt` → `audio_chunks/` |
| 2 | `02_transcribe_twi.py` | Twi speech recognition, one row per chunk | `audio_chunks/` → `metadata.csv` |
| 3 | `03_translate_to_english.py` | Twi → English with Gemini | `metadata.csv` → `english_translations.csv` |
| 4 | `04_push_speech_dataset.py` | *(optional)* publishes the audio + transcription corpus | → HF dataset |
| 5 | `05_push_parallel_dataset.py` | *(optional)* publishes the Twi–English corpus | → HF dataset |
| 6 | `06_generate_qa.py` | ~10 standalone farmer questions + answers per passage | `english_translations.csv` → `farmer_qa_output.jsonl` |
| 7 | `07_flatten_qa.py` | Explodes the nesting into one row per QA pair | `.jsonl` → `farmer_qa_flattened.csv` |
| 8 | `08_translate_qa.py` | Translates QA into Ewe, Ga, Twi | `flattened.csv` → `translated.csv` |
| 9 | `09_publish_datasets.py` | One dataset per language + generated cards, pushed to any number of orgs | `clean.csv` → HF datasets |
| 10 | `10_validate_output.py` | Gates publishing; drops bad rows with `--write-clean` | `translated.csv` → `clean.csv` |

Stage 10 runs *before* stage 9 in the orchestrator despite its number — validation
gates publishing.

## Models, and why they are pinned

Two stages call an LLM, and they call **Gemini specifically**:

| Stage | Model | Configuration | Job |
|---|---|---|---|
| 3 | `gemini-3.1-flash-lite` | thinking level HIGH, Google Search grounding on | Translate Twi → English |
| 6 | `gemini-3.6-flash` | default config | Write QA pairs from English passages |

Stage 8 is not an LLM at all — it uses Google Translate through the
`py-googletrans` fork.

Neither task is vendor-specific in principle — stage 3 is translation and stage 6
is generation, and any competent instruction-following model could attempt both.
An earlier revision of this repo made the provider swappable so that free
endpoints (NVIDIA NIM, any OpenAI-compatible server) could be used instead.

**That was removed deliberately.** The published datasets were produced with the
two models above, and those are the only models whose output on this task has
actually been inspected. A swappable provider invites someone to regenerate the
corpus with an untested model and get quietly worse Twi translations or
lower-quality questions, with nothing in the pipeline to catch it — stage 10
validates the *Ghanaian-language* translations, not the English QA or the
Twi→English step. Pinning the models is what makes the output reproducible and
the quality claim on the dataset cards defensible.

If you do want to swap in another model, change `--model` (stage 3) or
`MODEL_NAME` (stage 6) and **evaluate the output yourself before publishing**.
Don't assume parity.

## Two things that will bite you

**Google Translate uses `gaa` for Ga. Plain `ga` is Irish.** It returns fluent
Irish rather than erroring, so the mistake is silent and survives into a finished
dataset. This cost a full translation run once. `LANGS` in `08_translate_qa.py`
holds the correct mapping, and stage 10 checks the output for Irish function words.

**Pivot mode trades fidelity for nothing in most cases.** `--mode pivot` routes
`en → th → target`. It is supported because it was asked for, but direct mode
produces noticeably better Ewe and Twi. Prefer `--mode direct` unless you have a
specific reason not to.

## Translation modes

```bash
python3 pipeline/08_translate_qa.py --mode direct              # en -> ee/gaa/ak
python3 pipeline/08_translate_qa.py --mode pivot --pivot-lang th
```

Both validate every translation as non-empty and different from its source, with
retries. Neither guarantees correctness — see the limitations on the dataset cards.

## Published datasets

| Dataset | Content |
|---|---|
| [`ghanaopendata/ghana-farmer-qa`](https://huggingface.co/datasets/ghanaopendata/ghana-farmer-qa) | English original, 151,469 pairs |
| [`ghanaopendata/ghana-farmer-qa-ewe`](https://huggingface.co/datasets/ghanaopendata/ghana-farmer-qa-ewe) | English + Ewe |
| [`ghanaopendata/ghana-farmer-qa-ga`](https://huggingface.co/datasets/ghanaopendata/ghana-farmer-qa-ga) | English + Ga |
| [`ghanaopendata/ghana-farmer-qa-twi`](https://huggingface.co/datasets/ghanaopendata/ghana-farmer-qa-twi) | English + Twi |

Mirrored under [`ghananlpcommunity`](https://huggingface.co/ghananlpcommunity).
Upstream speech corpus:
[`ghananlpcommunity/twi-english-agric`](https://huggingface.co/datasets/ghananlpcommunity/twi-english-agric).

## Repository layout

```
pipeline/     the ten stages, numbered in execution order
prompts/      the QA generation prompt, with a worked example
samples/      ~100-row samples of each stage's output, to see the data shape
run_pipeline.sh
requirements.txt
```

Large artefacts — audio, JSONL, full CSVs, checkpoints — are gitignored. The
datasets live on Hugging Face; `samples/` is there so you can inspect the shape of
each stage without downloading or running anything.

## Credentials

Read from the environment, never committed:

- `GEMINI_API_KEY` — stages 3 and 6
- `HF_TOKEN`, or a cached `hf auth login` — stages 4, 5, 9

## Caveats on the data

The English answers are model-written from farmer interviews and **not reviewed by
an agronomist**; the Ghanaian-language text is **unreviewed machine translation**.
Neither is deployable extension advice as-is. `category` is free-form model output
with thousands of near-duplicate variants and needs clustering before use as a
label. Topic coverage follows whatever the source interviews happened to discuss.

Human post-editing is the highest-value next step, and the row layout — English and
target language side by side, with the source passage — is built for it.

## License

Code: MIT. Datasets: CC BY-NC 4.0, inherited from the upstream speech corpus.
