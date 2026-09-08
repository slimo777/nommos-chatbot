import json

import numpy as np

from embed import embed


def load_store(path):
    with open(path, encoding="utf-8") as f:
        store = json.load(f)
    texts = [item["text"] for item in store]
    matrix = np.array([item["vector"] for item in store])
    return texts, matrix


def top_hits(texts, matrix, qv, k=5):
    scores = matrix @ qv / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(qv))
    order = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), texts[i]) for i in order]


QUESTIONS = [
    "how much is the Nommos?",
    "what time do you close?",
    "do you have anything vegetarian?",
    "do you sell car tyres?",
]

if __name__ == "__main__":
    stores = {
        "category": load_store("embeddings_category.json"),
        "item": load_store("embeddings_item.json"),
    }

    for question in QUESTIONS:
        qv = np.array(embed(question, "RETRIEVAL_QUERY"))
        print(f"\n=== {question}")
        for name, (texts, matrix) in stores.items():
            hits = top_hits(texts, matrix, qv)
            spread = hits[0][0] - hits[-1][0]
            print(f"  [{name}]  top {hits[0][0]:.3f}   spread {spread:.3f}")
            for score, text in hits:
                print(f"      {score:.3f}  {text.splitlines()[0][:55]}")