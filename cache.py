"""
Semantic query cache for Pulsebot.

Embeds each query with all-MiniLM-L6-v2 (same model used for RAG).
On lookup, finds the nearest neighbour; if cosine similarity ≥ CACHE_SIMILARITY
the cached answer is returned immediately — zero RAG, zero Claude call.

Config:
  CACHE_ENABLED     true/false   (default: true)
  CACHE_SIMILARITY  0.0–1.0      (default: 0.92)
  CACHE_TTL         seconds      (default: 3600)
  CACHE_MAX         entries      (default: 500)
"""
import os
import time
import threading
from typing import Optional

import numpy as np

CACHE_ENABLED    = os.getenv("CACHE_ENABLED",    "true").lower() == "true"
CACHE_SIMILARITY = float(os.getenv("CACHE_SIMILARITY", "0.92"))
CACHE_TTL        = int(os.getenv("CACHE_TTL",    "3600"))
CACHE_MAX        = int(os.getenv("CACHE_MAX",    "500"))


class SemanticCache:
    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._vecs: list[np.ndarray] = []
        self._rows: list[dict]       = []
        self.hits   = 0
        self.misses = 0

    # ── embedding ──────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        from indexer import get_model
        emb  = get_model().encode([text], convert_to_numpy=True)[0].astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    # ── public API ─────────────────────────────────────────────────────────────

    def get(self, query: str, author: str) -> Optional[dict]:
        """Return cached entry if a sufficiently similar prior query exists."""
        if not CACHE_ENABLED or not self._vecs:
            self.misses += 1
            return None

        q_vec = self._embed(query)
        now   = time.time()

        with self._lock:
            best_sim, best_idx = -1.0, -1
            for i, (vec, row) in enumerate(zip(self._vecs, self._rows)):
                if row["author"] != author:
                    continue
                if now - row["ts"] > CACHE_TTL:
                    continue
                sim = float(np.dot(q_vec, vec))
                if sim > best_sim:
                    best_sim, best_idx = sim, i

            if best_idx >= 0 and best_sim >= CACHE_SIMILARITY:
                self.hits += 1
                return {**self._rows[best_idx], "cache_similarity": round(best_sim, 3)}

        self.misses += 1
        return None

    def put(self, query: str, author: str, answer: str,
            sources: list, confidence: float) -> None:
        if not CACHE_ENABLED:
            return
        q_vec = self._embed(query)
        row   = {
            "query": query, "author": author, "answer": answer,
            "sources": sources, "confidence": confidence, "ts": time.time(),
        }
        with self._lock:
            if len(self._vecs) >= CACHE_MAX:
                self._vecs.pop(0)
                self._rows.pop(0)
            self._vecs.append(q_vec)
            self._rows.append(row)

    def invalidate(self, author: str) -> None:
        with self._lock:
            pairs      = [(v, r) for v, r in zip(self._vecs, self._rows) if r["author"] != author]
            self._vecs = [p[0] for p in pairs]
            self._rows = [p[1] for p in pairs]

    # ── stats ──────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0


_cache = SemanticCache()
