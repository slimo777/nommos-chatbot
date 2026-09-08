import json

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


def search(question, k=3):
    q = np.array(embed(question, "RETRIEVAL_QUERY"))
    scores = matrix @ q / (norms * np.linalg.norm(q))
    order = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), texts[i]) for i in order]


if __name__ == "__main__":
    questions = [
        "how much is the Nommos?",
        "what time do you close?",
        "do you have anything vegetarian?",
        "combien coûte le homard ?",
        "do you sell car tyres?",
    ]
    for q in questions:
        print(f"\n=== {q}")
        for score, text in search(q, k=6):
            print(f"  {score:.3f}  {text.splitlines()[0][:55]}")