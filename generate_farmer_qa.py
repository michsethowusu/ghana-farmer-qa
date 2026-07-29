import os
import sys
import json
import time
import asyncio
import argparse
from typing import List, Dict, Any
from datasets import load_dataset
from google import genai
from pydantic import BaseModel, TypeAdapter

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-3.6-flash"

class QAPair(BaseModel):
    question: str
    knowledge: str

class CategoryQA(BaseModel):
    category: str
    qa_pairs: List[QAPair]

QA_TYPE_ADAPTER = TypeAdapter(List[CategoryQA])

PROMPT_TEMPLATE = """give me 10 simple general questions that farmers in Ghana are likely to have about crops and their behaviour based on the text below. questions that someone that doesn't know much asks and then they get to knowledge. the questions should not be depednent ofn eachother ie each stands on its own an shouldnt assume that it is a follow up of a previous question and should also not assume the the farmer is looking at the context text provided since the farmer does not have knowledge of the context.the questions should be in first person as if a farmer asking an extension officer questions.

Context text:
{context_text}

see sample response format:
[
  {{
    "category": "Basic Field Setup",
    "qa_pairs": [
      {{
        "question": "Why do I need to plant plantains next to my new cocoa trees?",
        "knowledge": "Young cocoa trees are like human babies—they burn easily under full sun. Plantains grow fast and have huge leaves that act as living umbrellas, giving cocoa the shade and moisture it needs to survive its first couple of years."
      }}
    ]
  }}
]

Return ONLY valid JSON matching the format above without any extra markdown formatting or conversational text if possible. (If using markdown ```json, ensure it's clean).
"""

def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

async def generate_qa_for_text_async(client: genai.Client, text: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    prompt = PROMPT_TEMPLATE.format(context_text=text)
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            raw_text = response.text
            cleaned = clean_json_response(raw_text)
            parsed_json = json.loads(cleaned)
            validated_data = QA_TYPE_ADAPTER.validate_python(parsed_json)
            return [cat.model_dump() for cat in validated_data]
        except Exception as e:
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 * attempt)
    return []

class RateLimiter:
    def __init__(self, requests_per_minute: float = 70.0):
        self.interval = 60.0 / requests_per_minute
        self.last_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed < self.interval:
                wait_time = self.interval - elapsed
                await asyncio.sleep(wait_time)
            self.last_time = time.time()

async def process_row(
    idx: int,
    row: Dict[str, Any],
    client: genai.Client,
    rate_limiter: RateLimiter,
    output_file: str,
    checkpoint_file: str,
    processed_indices: set,
    state_lock: asyncio.Lock
):
    file_name = row.get("file_name", f"row_{idx}")
    translated_text = row.get("translated_english", "")
    
    if not translated_text.strip():
        async with state_lock:
            processed_indices.add(idx)
        return "skipped_empty"

    await rate_limiter.acquire()
    
    try:
        qa_results = await generate_qa_for_text_async(client, translated_text)
        record = {
            "index": idx,
            "file_name": file_name,
            "original_twi": row.get("original_twi", ""),
            "translated_english": translated_text,
            "qa_data": qa_results
        }
        
        async with state_lock:
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_indices.add(idx)
            with open(checkpoint_file, "w", encoding="utf-8") as cp_f:
                json.dump({"processed_indices": list(processed_indices)}, cp_f)
        return "success"
    except Exception as e:
        print(f"ERROR: Row {idx} ({file_name}) failed: {e}", file=sys.stderr)
        async with state_lock:
            processed_indices.add(idx)
            with open(checkpoint_file, "w", encoding="utf-8") as cp_f:
                json.dump({"processed_indices": list(processed_indices)}, cp_f)
        return "error"

async def main_async():
    parser = argparse.ArgumentParser(description="Async Generate Farmer Q&A with Gemini")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset examples")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent request tasks")
    parser.add_argument("--rpm", type=float, default=70.0, help="Requests per minute target")
    parser.add_argument("--output", type=str, default="farmer_qa_output.jsonl", help="Output JSONL file")
    parser.add_argument("--checkpoint", type=str, default="farmer_qa_checkpoint.json", help="Checkpoint file")
    args = parser.parse_args()

    print(f"Loading dataset ghananlpcommunity/twi-english-agric...")
    dataset = load_dataset("ghananlpcommunity/twi-english-agric", split="train")
    
    client = genai.Client(api_key=API_KEY)
    
    processed_indices = set()
    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r", encoding="utf-8") as f:
                data = json.load(f)
                processed_indices = set(data.get("processed_indices", []))
            print(f"Resumed from checkpoint. Already processed: {len(processed_indices)} items.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.", file=sys.stderr)

    total_rows = len(dataset)
    end_index = total_rows if args.limit is None else min(total_rows, args.start_index + args.limit)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) if os.path.dirname(args.output) else '.', exist_ok=True)

    rate_limiter = RateLimiter(requests_per_minute=args.rpm)
    state_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)

    success_count = 0
    error_count = 0
    skipped_count = 0

    async def sem_process(idx, row):
        nonlocal success_count, error_count, skipped_count
        async with semaphore:
            if idx in processed_indices:
                return
            res = await process_row(
                idx, row, client, rate_limiter,
                args.output, args.checkpoint, processed_indices, state_lock
            )
            if res == "success":
                success_count += 1
                print(f"Processed row {idx} successfully.")
            elif res == "error":
                error_count += 1
                print(f"Row {idx} failed after retries (skipped).")
            else:
                skipped_count += 1

    tasks = []
    for idx in range(args.start_index, end_index):
        if idx in processed_indices:
            continue
        tasks.append(sem_process(idx, dataset[idx]))

    print(f"Starting async processing of {len(tasks)} items with concurrency={args.concurrency}, target ~{args.rpm} RPM...")
    await asyncio.gather(*tasks)

    print(f"Done! Success: {success_count}, Errors/Skipped: {error_count}, Empty/Skipped: {skipped_count}")
    print(f"Output saved to {args.output}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
