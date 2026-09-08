import json
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

from chunk import load_chunks

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-embedding-001"


def embed(text, task_type):
    result = client.models.embed_content(
        model=MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values


def embed_with_retry(text, task_type, tries=5):
    for attempt in range(tries):
        try:
            return embed(text, task_type)
        except errors.ClientError as e:
            if getattr(e, "code", None) != 429:
                raise
            wait = 12 * (attempt + 1)
            print(f"   rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("still rate limited after 5 tries")


if __name__ == "__main__":
    granularity = sys.argv[1] if len(sys.argv) > 1 else "item"
    chunks = load_chunks(granularity=granularity)
    out = f"embeddings_{granularity}.json"

    store = []
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            store = json.load(f)
        print(f"resuming - {len(store)} already done")

    done = {item["text"] for item in store}
    todo = [c for c in chunks if c not in done]
    print(f"{len(todo)} chunks to embed")

    try:
        for i, chunk in enumerate(todo, 1):
            vector = embed_with_retry(chunk, "RETRIEVAL_DOCUMENT")
            store.append({"text": chunk, "vector": vector})
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}")
            time.sleep(0.7)
    finally:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(store, f)
        print(f"saved {len(store)} vectors to {out}")