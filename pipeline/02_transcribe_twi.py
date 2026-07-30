#!/usr/bin/env python3
"""
Stage 2 -- Transcribe Twi audio chunks to text.

Transcribes every WAV in a directory with the Google Speech Recognition API
(language code "ak" = Akan/Twi) and writes file,transcription rows to a CSV.

Resumes safely:
- Chunks already present in the output CSV are skipped.
- Chunks that fail every attempt are appended to a failed-files list and skipped
  on later runs, so a permanently unintelligible chunk is not retried forever.

Requirements: SpeechRecognition, pydub.
"""
import argparse
import csv
import glob
import os
import sys
import time

import speech_recognition as sr


def transcribe_one(audio_path, language, max_attempts):
    """Return a transcription, or "" if every attempt fails."""
    recognizer = sr.Recognizer()
    for attempt in range(1, max_attempts + 1):
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            result = recognizer.recognize_google(audio, language=language)
            if result:
                return result
            print(f"   ! empty result (attempt {attempt}/{max_attempts})")
        except sr.UnknownValueError:
            print(f"   ! could not understand audio (attempt {attempt}/{max_attempts})")
        except sr.RequestError as e:
            print(f"   ! API error: {e} (attempt {attempt}/{max_attempts})")
        except Exception as e:
            print(f"   ! unexpected error: {e} (attempt {attempt}/{max_attempts})")
        if attempt < max_attempts:
            time.sleep(1)
    return ""


def load_lines(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main():
    p = argparse.ArgumentParser(description="Transcribe Twi audio chunks to CSV")
    p.add_argument("--input-dir", default="audio_chunks", help="Directory of WAV chunks")
    p.add_argument("--output", default="metadata.csv", help="Output CSV (file,transcription)")
    p.add_argument("--failed-list", default="failed_files.txt", help="Chunks that failed all attempts")
    p.add_argument("--language", default="ak", help="Speech API language code (ak = Akan/Twi)")
    p.add_argument("--pattern", default="*.wav", help="Filename glob")
    p.add_argument("--max-attempts", type=int, default=3, help="Attempts per chunk")
    p.add_argument("--recursive", action="store_true", help="Search subdirectories")
    args = p.parse_args()

    if args.recursive:
        files = glob.glob(os.path.join(args.input_dir, "**", args.pattern), recursive=True)
    else:
        files = glob.glob(os.path.join(args.input_dir, args.pattern))
    if not files:
        sys.exit(f"No files matching '{args.pattern}' in {args.input_dir}")
    files.sort()
    print(f"Found {len(files)} audio file(s).")

    transcriptions = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                transcriptions[row["file"]] = row["transcription"]
        print(f"Resuming: {len(transcriptions)} already transcribed.")

    failed = load_lines(args.failed_list)
    if failed:
        print(f"Skipping {len(failed)} previously failed chunk(s).")

    new_ok = 0
    new_failed = 0
    for idx, path in enumerate(files, 1):
        rel = os.path.relpath(path, args.input_dir)
        if rel in transcriptions or rel in failed:
            continue

        print(f"[{idx}/{len(files)}] {rel}")
        text = transcribe_one(path, args.language, args.max_attempts)

        if text:
            transcriptions[rel] = text
            new_ok += 1
            # Rewrite the whole CSV after each success so an interrupt never
            # leaves a half-written row.
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["file", "transcription"])
                w.writeheader()
                for fname, t in transcriptions.items():
                    w.writerow({"file": fname, "transcription": t})
            print(f"   => {text[:100]}{'...' if len(text) > 100 else ''}")
        else:
            failed.add(rel)
            new_failed += 1
            with open(args.failed_list, "w", encoding="utf-8") as f:
                for r in sorted(failed):
                    f.write(r + "\n")

    print(f"\nNew transcriptions: {new_ok}   new failures: {new_failed}")
    print(f"Total in {args.output}: {len(transcriptions)}")


if __name__ == "__main__":
    main()
