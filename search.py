import json
import time

import numpy as np

from embed import embed

STORES = ["embeddings_category.json", "embeddings_item.json"]

texts = []
vectors = []
for path in STORES:
    with open(path, encoding="utf-8") as f:
        for item in json.load(f):
            texts.append(item["text"])
            vectors.append(item["vector"])

matrix = np.array(vectors)
norms = np.linalg.norm(matrix, axis=1)


def search(question, k=3, tries=2):
    q = None
    for attempt in range(tries):
        try:
            q = np.array(embed(question, "RETRIEVAL_QUERY"))
            break
        except Exception as e:
            print(f"[search] embed failed: {type(e).__name__}: {e}")
            time.sleep(1.5 * (attempt + 1))

    if q is None:
        return []

    scores = matrix @ q / (norms * np.linalg.norm(q))
    order = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), texts[i]) for i in order]