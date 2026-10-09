#!/usr/bin/env python3
"""
Pulsebot — Per-Author FAISS Indexer (Piece 2)

Reads markdown articles from data/<author>/, chunks them, embeds with
sentence-transformers, and writes a FAISS index to indexes/<author>/.

Usage:
    python indexer.py --author lenny-rachitsky
    python indexer.py --author lenny-rachitsky --data-dir ./data --index-dir ./indexes
    python indexer.py --author lenny-rachitsky --rebuild   # force full re-index
"""

import re
import sys
import pickle
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import frontmatter
import faiss
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from models import Chunk  # stable import path for pickle compatibility

console = Console()

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_WORDS = 400
OVERLAP_WORDS = 50


# ── Text chunking ──────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


# ── Parse markdown with frontmatter ───────────────────────────────────────────

def _parse_markdown(path: Path, author: str) -> list[Chunk]:
    try:
        post = frontmatter.load(str(path))
    except Exception:
        return []

    text = post.content.strip()
    if not text:
        return []

    title = str(post.get("title", path.stem))
    url = str(post.get("url", ""))
    date = str(post.get("date", ""))
    fhash = _file_hash(path)

    chunks: list[Chunk] = []
    for i, chunk_text in enumerate(_chunk_text(text)):
        chunks.append(Chunk(
            author=author,
            source_file=str(path),
            title=title,
            url=url,
            date=date,
            text=chunk_text,
            chunk_index=i,
            file_hash=fhash,
        ))
    return chunks


# ── Load or build embedding model ─────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        console.print(f"[dim]Loading embedding model {MODEL_NAME}…[/dim]")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Index I/O ──────────────────────────────────────────────────────────────────

def _index_dir(base_dir: str, author: str) -> Path:
    p = Path(base_dir) / author
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_index(base_dir: str, author: str) -> tuple[Optional[faiss.Index], list[Chunk]]:
    d = _index_dir(base_dir, author)
    idx_path = d / "index.faiss"
    chunks_path = d / "chunks.pkl"
    if idx_path.exists() and chunks_path.exists():
        idx = faiss.read_index(str(idx_path))
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return idx, chunks
    return None, []


def save_index(base_dir: str, author: str, idx: faiss.Index, chunks: list[Chunk]) -> None:
    d = _index_dir(base_dir, author)
    faiss.write_index(idx, str(d / "index.faiss"))
    with open(d / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    console.print(f"[green]✓ Saved index: {d} ({len(chunks)} chunks)[/green]")


# ── Build/update index for one author ─────────────────────────────────────────

def build_index(
    author: str,
    data_dir: str = "./data",
    index_dir: str = "./indexes",
    rebuild: bool = False,
) -> tuple[faiss.Index, list[Chunk]]:
    """
    Build (or incrementally update) the FAISS index for one author.
    Returns (index, chunks).
    """
    md_dir = Path(data_dir) / author
    if not md_dir.exists():
        console.print(f"[red]No data directory for author '{author}': {md_dir}[/red]")
        sys.exit(1)

    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        console.print(f"[yellow]No markdown files found in {md_dir}[/yellow]")
        sys.exit(1)

    console.print(f"[dim]Found {len(md_files)} markdown files for {author}[/dim]")

    # Load existing index (for incremental updates)
    existing_idx, existing_chunks = (None, []) if rebuild else load_index(index_dir, author)

    # Track which file hashes are already indexed
    indexed_hashes: set[str] = {c.file_hash for c in existing_chunks}

    # Parse only new/changed files
    new_chunks: list[Chunk] = []
    skipped = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing articles", total=len(md_files))
        for path in md_files:
            fhash = _file_hash(path)
            if fhash in indexed_hashes:
                skipped += 1
            else:
                new_chunks.extend(_parse_markdown(path, author))
            progress.advance(task)

    if skipped:
        console.print(f"[dim]Skipped {skipped} unchanged files.[/dim]")

    if not new_chunks and existing_idx is not None:
        console.print(f"[green]Index is up-to-date. {len(existing_chunks)} chunks.[/green]")
        return existing_idx, existing_chunks

    if not new_chunks:
        console.print("[red]No content to index.[/red]")
        sys.exit(1)

    # Embed new chunks
    model = get_model()
    console.print(f"[dim]Embedding {len(new_chunks)} new chunks…[/dim]")
    texts = [c.text for c in new_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # L2-normalize for cosine similarity via IndexFlatIP
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    if existing_idx is not None and not rebuild:
        # Append to existing index
        existing_idx.add(embeddings)
        all_chunks = existing_chunks + new_chunks
    else:
        idx = faiss.IndexFlatIP(dim)
        idx.add(embeddings)
        all_chunks = new_chunks

        # If we skipped files but are rebuilding, need all existing chunks too
        # (rebuild=True means we already cleared existing_chunks above)
        if rebuild and skipped:
            # We set existing_chunks=[] for rebuild, so re-parse all files
            pass

        existing_idx = idx

    save_index(index_dir, author, existing_idx, all_chunks)
    return existing_idx, all_chunks


# ── Smoke-test search ──────────────────────────────────────────────────────────

def search(query: str, author: str, index_dir: str = "./indexes", top_k: int = 5) -> list[dict]:
    """Search the index for an author. Returns top-k results."""
    idx, chunks = load_index(index_dir, author)
    if idx is None:
        return []

    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)

    scores, indices = idx.search(q_emb, top_k)
    results = []
    for score, i in zip(scores[0], indices[0]):
        if i < 0:
            continue
        c = chunks[i]
        results.append({
            "score": float(score),
            "title": c.title,
            "url": c.url,
            "date": c.date,
            "chunk_index": c.chunk_index,
            "text": c.text[:300],
        })
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-author FAISS index from markdown articles.")
    parser.add_argument("--author",    required=True, help="Author handle (must match data/<author>/ directory)")
    parser.add_argument("--data-dir",  default="./data",    help="Root directory containing author markdown folders")
    parser.add_argument("--index-dir", default="./indexes", help="Root directory to write indexes into")
    parser.add_argument("--rebuild",   action="store_true",  help="Force full rebuild instead of incremental update")
    parser.add_argument("--query",     help="Optional test query to run after building")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]Pulsebot Indexer[/bold cyan]  author=[cyan]{args.author}[/cyan]  rebuild={args.rebuild}\n")

    idx, chunks = build_index(
        author=args.author,
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        rebuild=args.rebuild,
    )

    console.print(f"\n[bold green]Done.[/bold green] Index has {idx.ntotal} vectors across {len(chunks)} chunks.\n")

    if args.query:
        console.print(f"[bold]Query:[/bold] {args.query}\n")
        results = search(args.query, args.author, args.index_dir)
        for i, r in enumerate(results, 1):
            console.print(f"[bold]{i}.[/bold] [cyan]{r['title']}[/cyan]  score={r['score']:.3f}")
            console.print(f"   {r['text'][:150]}…\n")


if __name__ == "__main__":
    main()
