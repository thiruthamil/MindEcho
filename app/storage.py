# app/storage.py — FAISS + Ollama embeddings
from __future__ import annotations
from pathlib import Path
import os
import uuid
from typing import List, Tuple

import numpy as np
import faiss
import requests

OLLAMA_NATIVE_BASE = os.getenv("OLLAMA_NATIVE_BASE", "http://localhost:11434")


def _embed_one(model: str, text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_NATIVE_BASE}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json() or {}

    if isinstance(data.get("embedding"), list):
        return np.array(data["embedding"], dtype=np.float32)

    if isinstance(data.get("embeddings"), list) and data["embeddings"]:
        return np.array(data["embeddings"][0], dtype=np.float32)

    raise RuntimeError(f"Unexpected /api/embeddings payload: {list(data.keys())}")


def _ollama_embed(model: str, texts: List[str]) -> np.ndarray:
    vecs = [_embed_one(model, t) for t in texts]
    dims = {v.shape[0] for v in vecs}
    if len(dims) != 1:
        raise RuntimeError(f"Inconsistent embedding dims: {dims}")
    return np.stack(vecs, axis=0).astype(np.float32, copy=False)


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


class VectorStore:
    """Cosine similarity via FAISS inner product on normalized vectors."""

    def __init__(self, data_dir: str, model_name: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.data_dir / "vectors.faiss"
        self.id_path = self.data_dir / "idmap.npy"

        self.model_name = model_name or "nomic-embed-text"
        self.index: faiss.IndexFlatIP | None = None
        self.idmap: np.ndarray | None = None
        self.dim: int | None = None

        self._load()

        # Fresh store: probe to get dimension, build empty index
        if self.index is None:
            probe = _ollama_embed(self.model_name, ["__dim_probe__"])
            self.dim = int(probe.shape[1])
            self.index = faiss.IndexFlatIP(self.dim)
            self.idmap = np.empty((0,), dtype=np.int64)

    def _load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dim = self.index.d
            self.idmap = np.load(self.id_path) if self.id_path.exists() else np.arange(self.index.ntotal)
        else:
            self.index = None
            self.idmap = None

    def _save(self) -> None:
        if self.index is None:
            raise RuntimeError("Index not initialized")
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.id_path, self.idmap if self.idmap is not None else np.empty((0,), dtype=np.int64))

    def embed(self, texts: List[str]) -> np.ndarray:
        return _normalize(_ollama_embed(self.model_name, texts))

    def add(self, text: str) -> Tuple[int, str]:
        if self.index is None:
            probe = _ollama_embed(self.model_name, ["__dim_probe__"])
            self.dim = int(probe.shape[1])
            self.index = faiss.IndexFlatIP(self.dim)
            self.idmap = np.empty((0,), dtype=np.int64)

        vec = self.embed([text])
        before = self.index.ntotal
        self.index.add(vec)
        faiss_id = before

        if self.idmap is None or self.idmap.size == 0:
            self.idmap = np.array([faiss_id], dtype=np.int64)
        else:
            self.idmap = np.concatenate([self.idmap, np.array([faiss_id], dtype=np.int64)])

        self._save()
        return faiss_id, str(uuid.uuid4())

    def search(self, text: str, k: int = 3):
        if self.index is None or self.index.ntotal == 0:
            return [], []
        q = self.embed([text])
        scores, idx = self.index.search(q, k)
        return idx[0].tolist(), scores[0].tolist()
