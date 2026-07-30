#!/usr/bin/env python3
"""
Stage 1 -- Download YouTube audio and split it into fixed-length WAV chunks.

Downloads the audio track of every URL in a list, converts it to 16 kHz mono WAV,
and slices it into fixed-length chunks named <videoID>_<NNN>.wav. Short trailing
chunks are discarded so every chunk is exactly CHUNK_SEC long.

Resumes safely: completed URLs are recorded in <output-dir>/completed_urls.txt and
skipped on later runs.

Requirements: yt-dlp, pydub, ffmpeg on PATH.
"""
import argparse
import os
import sys

import yt_dlp
from pydub import AudioSegment
from pydub.utils import make_chunks


def main():
    p = argparse.ArgumentParser(description="Download YouTube audio and chunk it into WAVs")
    p.add_argument("--urls", default="urls.txt", help="Text file with one YouTube URL per line")
    p.add_argument("--output-dir", default="audio_chunks", help="Where to write WAV chunks")
    p.add_argument("--chunk-sec", type=int, default=30, help="Chunk length in seconds")
    p.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate (Hz)")
    p.add_argument("--keep-full", action="store_true", help="Keep the full downloaded WAV")
    args = p.parse_args()

    if not os.path.exists(args.urls):
        sys.exit(f"URL file not found: {args.urls}")

    os.makedirs(args.output_dir, exist_ok=True)
    chunk_len_ms = args.chunk_sec * 1000

    completed_file = os.path.join(args.output_dir, "completed_urls.txt")
    completed = set()
    if os.path.exists(completed_file):
        with open(completed_file, "r", encoding="utf-8") as f:
            completed = {line.strip() for line in f if line.strip()}

    with open(args.urls, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    to_process = [u for u in urls if u not in completed]
    print(f"URLs in file: {len(urls)}  already done: {len(completed)}  to process: {len(to_process)}")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ar", str(args.sample_rate), "-ac", "1"],
        "outtmpl": os.path.join(args.output_dir, "%(title)s.%(ext)s"),
        "quiet": False,
        "nooverwrites": True,
    }

    n_chunks = 0
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in to_process:
            try:
                print(f"\nDownloading: {url}")
                info = ydl.extract_info(url, download=True)

                video_id = info.get("id") or "unknown_video"

                wav_path = None
                for dl in info.get("requested_downloads", []):
                    if dl["filepath"].lower().endswith(".wav"):
                        wav_path = dl["filepath"]
                        break
                if not wav_path:
                    print(f"  Could not find WAV for {url}")
                    continue

                audio = AudioSegment.from_wav(wav_path)
                for i, chunk in enumerate(make_chunks(audio, chunk_len_ms)):
                    if len(chunk) != chunk_len_ms:
                        print(f"  Discarding short trailing chunk ({len(chunk) / 1000:.1f}s)")
                        continue
                    # <videoID>_<NNN>.wav keeps every chunk traceable to its source video
                    chunk_path = os.path.join(args.output_dir, f"{video_id}_{i + 1:03d}.wav")
                    chunk.export(chunk_path, format="wav")
                    n_chunks += 1

                if not args.keep_full:
                    os.remove(wav_path)

                with open(completed_file, "a", encoding="utf-8") as f:
                    f.write(url + "\n")
                completed.add(url)
                print(f"  done -- {n_chunks} chunks written so far")

            except Exception as e:
                print(f"Error processing {url}: {e}", file=sys.stderr)

    print(f"\nWrote {n_chunks} chunks to {args.output_dir}")


if __name__ == "__main__":
    main()
