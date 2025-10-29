# lab/embed_and_index.py
from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def main() -> None:
    notes = [
        "felt stuck on cnn homework, watched a 10-min explainer and it clicked",
        "nervous before interview, mock practice for 20 mins helped",
        "productive morning after planning tasks at night",
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    X = model.encode(notes, convert_to_numpy=True)
    X = _normalize(X)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    q = "anxious before interview, what helped"
    qv = model.encode([q], convert_to_numpy=True)
    qv = _normalize(qv)

    scores, ids = index.search(qv, 3)

    print("query:", q)
    for s, i in zip(scores[0], ids[0]):
        print(f"score={s:.2f}", notes[i])


if __name__ == "__main__":
    main()
