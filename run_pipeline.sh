#!/usr/bin/env bash
#
# Ghana Farmer QA -- end-to-end pipeline runner.
#
# Runs YouTube URLs -> audio -> Twi transcripts -> English -> QA pairs ->
# Ghanaian-language translations -> published Hugging Face datasets.
#
# Every stage resumes from where it left off, so re-running after an interrupt is
# safe and cheap. Stages 4 and 5 (publishing the intermediate speech corpora) are
# skipped unless you set PUSH_SPEECH=1.
#
# Usage:
#   ./run_pipeline.sh                  # run every stage
#   ./run_pipeline.sh 6 10             # run only stages 6 through 10
#   ORGS=myorg ./run_pipeline.sh 9 9   # publish only
#
set -euo pipefail

FROM_STAGE="${1:-1}"
TO_STAGE="${2:-10}"

# ---- configuration (override via environment) --------------------------------
URLS="${URLS:-urls.txt}"
AUDIO_DIR="${AUDIO_DIR:-audio_chunks}"
CHUNK_SEC="${CHUNK_SEC:-30}"
METADATA="${METADATA:-metadata.csv}"
ENGLISH="${ENGLISH:-english_translations.csv}"
QA_JSONL="${QA_JSONL:-farmer_qa_output.jsonl}"
FLAT="${FLAT:-farmer_qa_flattened.csv}"
MODE="${MODE:-direct}"                 # direct | pivot
PIVOT_LANG="${PIVOT_LANG:-th}"
TRANSLATED="${TRANSLATED:-}"           # defaults per MODE below
CLEAN="${CLEAN:-farmer_qa_clean.csv}"
ORGS="${ORGS:-}"                       # comma-separated HF orgs for stage 9
SPEECH_REPO="${SPEECH_REPO:-}"         # e.g. myorg/twi-agriculture-asr
PARALLEL_REPO="${PARALLEL_REPO:-}"     # e.g. myorg/twi-english-agric
PUSH_SPEECH="${PUSH_SPEECH:-0}"
CONCURRENCY="${CONCURRENCY:-15}"
BATCH_SIZE="${BATCH_SIZE:-30}"

if [[ -z "$TRANSLATED" ]]; then
  if [[ "$MODE" == "pivot" ]]; then
    TRANSLATED="farmer_qa_pivot_translated.csv"
  else
    TRANSLATED="farmer_qa_translated.csv"
  fi
fi

P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline"

run_stage() {  # $1 = stage number, rest = command
  local n="$1"; shift
  if (( n < FROM_STAGE || n > TO_STAGE )); then
    echo "-- skipping stage $n"
    return 0
  fi
  echo
  echo "=============================================================="
  echo "== Stage $n"
  echo "=============================================================="
  "$@"
}

# ---- preflight ---------------------------------------------------------------
if (( FROM_STAGE <= 6 && TO_STAGE >= 3 )) && [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY is not set -- stages 3 and 6 need it." >&2
  exit 1
fi

run_stage 1 python3 -u "$P/01_download_youtube_audio.py" \
  --urls "$URLS" --output-dir "$AUDIO_DIR" --chunk-sec "$CHUNK_SEC"

run_stage 2 python3 -u "$P/02_transcribe_twi.py" \
  --input-dir "$AUDIO_DIR" --output "$METADATA"

run_stage 3 python3 -u "$P/03_translate_to_english.py" \
  --input "$METADATA" --output "$ENGLISH"

if [[ "$PUSH_SPEECH" == "1" ]]; then
  [[ -n "$SPEECH_REPO" ]] && run_stage 4 python3 -u "$P/04_push_speech_dataset.py" \
    --repo-id "$SPEECH_REPO" --metadata "$METADATA" --audio-dir "$AUDIO_DIR"
  [[ -n "$PARALLEL_REPO" ]] && run_stage 5 python3 -u "$P/05_push_parallel_dataset.py" \
    --repo-id "$PARALLEL_REPO" --input "$ENGLISH" --chunk-sec "$CHUNK_SEC" --drop-failed
else
  echo "-- skipping stages 4-5 (set PUSH_SPEECH=1 with SPEECH_REPO / PARALLEL_REPO to publish)"
fi

run_stage 6 python3 -u "$P/06_generate_qa.py" --output "$QA_JSONL"

run_stage 7 python3 -u "$P/07_flatten_qa.py" --input "$QA_JSONL" --output "$FLAT"

run_stage 8 python3 -u "$P/08_translate_qa.py" \
  --mode "$MODE" --pivot-lang "$PIVOT_LANG" --input "$FLAT" --output "$TRANSLATED" \
  --batch-size "$BATCH_SIZE" --concurrency "$CONCURRENCY"

# Stage 10 gates stage 9: bad translations must never reach the Hub. It exits
# non-zero when it finds problems, and --write-clean drops the offending rows.
run_stage 10 python3 -u "$P/10_validate_output.py" \
  --input "$TRANSLATED" --write-clean "$CLEAN" || {
    echo
    echo "Validation found problems. Bad rows were dropped into $CLEAN."
    echo "Review that file, then run stage 9 against it."
  }

if (( FROM_STAGE <= 9 && TO_STAGE >= 9 )); then
  if [[ -z "$ORGS" ]]; then
    echo "-- stage 9: ORGS not set, building locally without publishing"
    run_stage 9 python3 -u "$P/09_publish_datasets.py" \
      --input "${CLEAN}" --build-only --orgs placeholder
  else
    PIVOT_ARG=()
    [[ "$MODE" == "pivot" ]] && PIVOT_ARG=(--pivot-lang "$PIVOT_LANG")
    run_stage 9 python3 -u "$P/09_publish_datasets.py" \
      --input "${CLEAN}" --orgs "$ORGS" "${PIVOT_ARG[@]}"
  fi
fi

echo
echo "Pipeline finished."
