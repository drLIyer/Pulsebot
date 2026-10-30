#!/usr/bin/env python3
"""
Pulsebot Evals

Metrics per question:
  confidence        — retrieval confidence (sigmoid of top cross-encoder score, 0–1)
  accuracy          — cosine sim / keyword hit-rate / LLM judge vs expected answer
  faithfulness      — fraction of answer claims supported by retrieved context
  context_precision — fraction of retrieved chunks that were actually useful
  context_recall    — were retrieved chunks sufficient to answer the question?
  answer_relevance  — does the answer directly address the question?
  abstention        — when confidence < 50%, did Claude hedge appropriately?
  latency           — TTFT / total wall time; p50/p95/p99 with --runs N
  cost              — estimated API cost per query from usage tokens

Usage:
    python evals.py --author lenny-rachitsky
    python evals.py --author lenny-rachitsky --runs 5
    python evals.py --author lenny-rachitsky --baseline evals/lenny-rachitsky/report.json
    python evals.py --author lenny-rachitsky --llm-judge
    python evals.py --author lenny-rachitsky --skip-llm     # no extra API calls
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── Thresholds ────────────────────────────────────────────────────────────────

CONF_WARN  = 0.50   # retrieval confidence — below this the answer may be unreliable
ACC_GOOD   = 0.75
ACC_OK     = 0.50
FAITH_GOOD = 0.80   # faithfulness: fraction of claims grounded in retrieved chunks
FAITH_OK   = 0.50
PREC_GOOD  = 0.75   # context precision: fraction of chunks that were useful
PREC_OK    = 0.50
REC_GOOD   = 0.80   # context recall: were chunks sufficient to answer?
REC_OK     = 0.50
REL_GOOD   = 0.80   # answer relevance: does answer address the question?
REL_OK     = 0.60

# ── Model pricing ($/1M tokens, approximate — update as prices change) ────────

MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-haiku-4-5":          {"input": 0.80,  "output": 4.00},
    "claude-sonnet-5-5":         {"input": 3.00,  "output": 15.00},
    "claude-opus-5-5":           {"input": 15.00, "output": 75.00},
}
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}

# Phrases that indicate appropriate hedging when the newsletter doesn't cover a topic
_HEDGE_PHRASES = [
    "don't have", "doesn't cover", "not covered", "can't find", "don't know",
    "not sure", "unable to find", "no information", "not discussed",
    "not mentioned", "not addressed", "not enough", "limited information",
    "verify", "primary source", "cannot find", "not available in",
    "newsletter doesn't", "i'm not", "cannot answer", "outside the scope",
]


# ── SSE client ────────────────────────────────────────────────────────────────

def _stream_chat(server: str, author: str, question: str) -> dict:
    """
    POST /chat/{author} with include_context=True and collect all SSE events.

    Returns:
        answer, sources, confidence, context (str), chunks (list),
        ttft, total, usage  — plus ttft_samples/total_samples if populated
        by _stream_chat_n.
    """
    url = f"{server.rstrip('/')}/chat/{author}"
    payload = {"query": question, "history": [], "include_context": True}

    answer_parts: list[str] = []
    sources: list[dict] = []
    confidence: Optional[float] = None
    context: Optional[str] = None
    chunks: list[dict] = []
    usage: dict = {}

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    total: float = 0.0

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "sources":
                    sources = event.get("sources", [])
                elif etype == "confidence":
                    confidence = event.get("score")
                elif etype == "context":
                    context = event.get("text")
                    chunks = event.get("chunks", [])
                elif etype == "token":
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    answer_parts.append(event.get("text", ""))
                elif etype == "done":
                    total = time.perf_counter() - t0
                    usage = event.get("usage", {})
                elif etype == "error":
                    raise RuntimeError(event.get("message", "unknown error"))

    return {
        "answer":     "".join(answer_parts),
        "sources":    sources,
        "confidence": confidence,
        "context":    context,
        "chunks":     chunks,
        "ttft":       ttft or total,
        "total":      total,
        "usage":      usage,
    }


def _stream_chat_n(server: str, author: str, question: str, n: int) -> dict:
    """Run the same question n times; returns first result with latency samples attached."""
    ttft_samples: list[float] = []
    total_samples: list[float] = []
    first: Optional[dict] = None
    for _ in range(n):
        r = _stream_chat(server, author, question)
        ttft_samples.append(r["ttft"])
        total_samples.append(r["total"])
        if first is None:
            first = r
    assert first is not None
    first["ttft_samples"]  = ttft_samples
    first["total_samples"] = total_samples
    return first


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# ── Shared LLM helper ─────────────────────────────────────────────────────────

def _haiku(prompt: str, max_tokens: int = 300) -> Optional[dict]:
    """Call Claude Haiku and extract a JSON object from the response."""
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return json.loads(text)
    except Exception as e:
        console.print(f"[dim yellow]Haiku scorer error: {e}[/dim yellow]")
        return None


# ── Accuracy scorers ──────────────────────────────────────────────────────────

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        console.print("[dim]Loading embedding model for accuracy scoring…[/dim]")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def score_accuracy_cosine(expected: str, actual: str) -> float:
    model = _get_embed_model()
    embs = model.encode([expected, actual], convert_to_numpy=True)
    return _cosine(embs[0], embs[1])


def score_accuracy_keywords(keywords: list[str], actual: str) -> float:
    if not keywords:
        return 1.0
    low = actual.lower()
    return sum(1 for kw in keywords if kw.lower() in low) / len(keywords)


def score_accuracy_llm(question: str, expected: str, actual: str) -> float:
    result = _haiku(
        "Score the ACTUAL answer against the EXPECTED answer for the QUESTION. "
        "Reply ONLY with JSON: {\"score\": <int 0-100>}\n\n"
        f"QUESTION: {question}\n\nEXPECTED: {expected}\n\nACTUAL: {actual}",
        max_tokens=30,
    )
    if result and "score" in result:
        return float(result["score"]) / 100.0
    return score_accuracy_cosine(expected, actual)


# ── Faithfulness ──────────────────────────────────────────────────────────────

def score_faithfulness(answer: str, context: str) -> Optional[float]:
    """
    Fraction of the answer's factual claims that are directly supported by
    the retrieved context.  Catches hallucination beyond the source material.
    """
    if not answer.strip() or not context.strip():
        return None
    result = _haiku(
        "You are evaluating a RAG chatbot for hallucination.\n\n"
        "Given the CONTEXT (retrieved newsletter excerpts) and the ANSWER, "
        "determine what fraction of the ANSWER's factual claims are directly "
        "supported by the CONTEXT.\n\n"
        "Steps:\n"
        "1. List each distinct factual claim in the ANSWER.\n"
        "2. Mark each claim supported (S) or unsupported (U) by the CONTEXT.\n"
        "3. Reply ONLY with JSON on the last line: "
        "{\"supported\": <int>, \"total\": <int>}\n\n"
        f"CONTEXT:\n{context[:4000]}\n\nANSWER:\n{answer}",
        max_tokens=400,
    )
    if result and "total" in result and result["total"] > 0:
        return result["supported"] / result["total"]
    return None


# ── Context precision ─────────────────────────────────────────────────────────

def score_context_precision(question: str, chunks: list[dict]) -> Optional[float]:
    """
    Fraction of retrieved chunks that were actually useful for answering.
    Low precision = retrieval returned too much noise.
    """
    if not chunks:
        return None
    chunk_list = "\n\n".join(
        f"[Chunk {i+1}] {c.get('title','')}: {c.get('text','')[:400]}"
        for i, c in enumerate(chunks)
    )
    result = _haiku(
        f"QUESTION: {question}\n\n"
        f"RETRIEVED CHUNKS:\n{chunk_list[:3500]}\n\n"
        "Rate each chunk's usefulness for answering the question:\n"
        "  2 = essential / directly relevant\n"
        "  1 = somewhat useful\n"
        "  0 = not useful / off-topic\n"
        f'Reply ONLY with JSON: {{"ratings": [<int>, ...]}}  '
        f"(exactly {len(chunks)} integers)",
        max_tokens=100,
    )
    if result and "ratings" in result:
        ratings = result["ratings"][:len(chunks)]
        if ratings:
            return sum(1 for r in ratings if r > 0) / len(ratings)
    return None


# ── Context recall ────────────────────────────────────────────────────────────

def score_context_recall(question: str, answer: str, chunks: list[dict]) -> Optional[float]:
    """
    Were the retrieved chunks sufficient to fully answer the question?
    Low recall = key information was missing from what was retrieved.
    """
    if not chunks:
        return None
    context = "\n\n".join(
        f"[{c.get('title','')}]: {c.get('text','')[:500]}"
        for c in chunks
    )
    result = _haiku(
        f"QUESTION: {question}\n\n"
        f"RETRIEVED CHUNKS:\n{context[:4000]}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Were the retrieved chunks sufficient to fully answer the question? "
        "Or does the answer seem incomplete because key information was missing from retrieval?\n"
        'Reply ONLY with JSON: {"verdict": "full" | "partial" | "insufficient"}',
    )
    if result and "verdict" in result:
        return {"full": 1.0, "partial": 0.5, "insufficient": 0.0}.get(result["verdict"])
    return None


# ── Answer relevance ──────────────────────────────────────────────────────────

def score_answer_relevance(question: str, answer: str) -> Optional[float]:
    """
    Does the answer directly and completely address the question?
    Catches evasive or off-topic answers that are otherwise grounded and accurate.
    """
    if not answer.strip():
        return 0.0
    result = _haiku(
        f"QUESTION: {question}\n\nANSWER: {answer}\n\n"
        "Score how directly and completely this answer addresses the question (0–100). "
        "A high score means the answer is on-topic and complete. "
        'Reply ONLY with JSON: {"score": <int 0-100>}',
        max_tokens=30,
    )
    if result and "score" in result:
        return float(result["score"]) / 100.0
    return None


# ── Abstention quality ────────────────────────────────────────────────────────

def score_abstention(answer: str, confidence: Optional[float]) -> Optional[float]:
    """
    When confidence < 50%, did Claude appropriately hedge instead of hallucinating?
    Returns 1.0 (correctly hedged), 0.0 (overconfident), None (not applicable).

    This is the most dangerous failure mode: low retrieval confidence but a
    confident-sounding answer with no caveats.
    """
    if confidence is None or confidence >= CONF_WARN:
        return None  # only meaningful at low confidence
    answer_low = answer.lower()
    hedged = any(phrase in answer_low for phrase in _HEDGE_PHRASES)
    return 1.0 if hedged else 0.0


# ── Cost ──────────────────────────────────────────────────────────────────────

def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# ── Eval runner ───────────────────────────────────────────────────────────────

def run_eval(
    server:        str,
    author:        str,
    golden:        list[dict],
    use_llm_judge: bool = False,
    skip_llm:      bool = False,
    n_runs:        int  = 1,
) -> list[dict]:
    results = []
    for i, item in enumerate(golden, 1):
        q = item["question"]
        console.print(f"[dim]  [{i}/{len(golden)}] {q[:72]}…[/dim]")
        try:
            r = _stream_chat_n(server, author, q, n_runs) if n_runs > 1 else _stream_chat(server, author, q)
        except Exception as e:
            console.print(f"[red]    Error: {e}[/red]")
            results.append({**item, "error": str(e)})
            continue

        answer   = r["answer"]
        context  = r.get("context") or ""
        chunks   = r.get("chunks", [])
        conf     = r.get("confidence")

        # ── Accuracy ──────────────────────────────────────────────────────────
        accuracy: Optional[float] = None
        acc_method = "—"
        if item.get("expected_answer"):
            if use_llm_judge and not skip_llm:
                accuracy = score_accuracy_llm(q, item["expected_answer"], answer)
                acc_method = "llm"
            else:
                accuracy = score_accuracy_cosine(item["expected_answer"], answer)
                acc_method = "cosine"
        elif item.get("expected_keywords"):
            accuracy = score_accuracy_keywords(item["expected_keywords"], answer)
            acc_method = "keywords"

        # ── LLM-based quality metrics ─────────────────────────────────────────
        faithfulness       = None
        context_precision  = None
        context_recall     = None
        answer_relevance   = None

        if not skip_llm and answer:
            console.print("[dim]    → LLM scorers…[/dim]")
            if context:
                faithfulness = score_faithfulness(answer, context)
            if chunks:
                context_precision = score_context_precision(q, chunks)
                context_recall    = score_context_recall(q, answer, chunks)
            answer_relevance = score_answer_relevance(q, answer)

        # ── Abstention (no LLM call) ──────────────────────────────────────────
        abstention = score_abstention(answer, conf)

        # ── Cost ──────────────────────────────────────────────────────────────
        usage = r.get("usage", {})
        cost  = compute_cost(
            usage.get("model", ""),
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

        results.append({
            **item,
            "answer":            answer,
            "sources":           r["sources"],
            "confidence":        conf,
            "ttft":              r["ttft"],
            "total":             r["total"],
            "ttft_samples":      r.get("ttft_samples", [r["ttft"]]),
            "total_samples":     r.get("total_samples", [r["total"]]),
            "usage":             usage,
            "cost":              cost,
            "accuracy":          accuracy,
            "acc_method":        acc_method,
            "faithfulness":      faithfulness,
            "context_precision": context_precision,
            "context_recall":    context_recall,
            "answer_relevance":  answer_relevance,
            "abstention":        abstention,
        })

    return results


# ── Report helpers ────────────────────────────────────────────────────────────

def _fmt(v: Optional[float], good: float, ok: float, pct: bool = False) -> str:
    if v is None:
        return "[dim]n/a[/dim]"
    label = f"{v*100:.0f}%" if pct else f"{v:.2f}"
    if v >= good:
        return f"[green]{label} ✓[/green]"
    if v >= ok:
        return f"[yellow]{label} ~[/yellow]"
    return f"[red]{label} ✗[/red]"

def _conf_fmt(v: Optional[float]) -> str:
    if v is None:
        return "[dim]n/a[/dim]"
    pct = v * 100
    return f"[green]{pct:.0f}%[/green]" if v >= CONF_WARN else f"[red]{pct:.0f}% ⚠[/red]"

def _abst_fmt(v: Optional[float]) -> str:
    if v is None:
        return "[dim]—[/dim]"
    return "[green]hedged ✓[/green]" if v == 1.0 else "[red]overconf ✗[/red]"

def _avg(vals: list[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


# ── Main report ───────────────────────────────────────────────────────────────

def print_report(author: str, results: list[dict]) -> None:
    # ── Table 1: core quality metrics ─────────────────────────────────────────
    t1 = Table(
        title=f"[bold cyan]{author}[/bold cyan] — Quality Metrics",
        box=box.ROUNDED, show_footer=True, expand=True,
    )
    t1.add_column("#",           width=3,  no_wrap=True)
    t1.add_column("Question",    ratio=3,  overflow="fold")
    t1.add_column("Conf%",       width=9,  justify="right", footer="Avg")
    t1.add_column("Accuracy",    width=12, justify="right", footer="Avg")
    t1.add_column("Faithful",    width=11, justify="right", footer="Avg")
    t1.add_column("Relevance",   width=11, justify="right", footer="Avg")
    t1.add_column("Abstention",  width=13, justify="right")

    cols: dict[str, list] = {k: [] for k in ["confidence","accuracy","faithfulness","answer_relevance"]}
    for i, r in enumerate(results, 1):
        if "error" in r:
            t1.add_row(str(i), r["question"][:80], "[red]ERR[/red]", "—", "—", "—", "—")
            continue
        for k in cols:
            if r.get(k) is not None:
                cols[k].append(r[k])
        t1.add_row(
            str(i), r["question"][:80],
            _conf_fmt(r.get("confidence")),
            _fmt(r.get("accuracy"),         ACC_GOOD,   ACC_OK),
            _fmt(r.get("faithfulness"),     FAITH_GOOD, FAITH_OK,  pct=True),
            _fmt(r.get("answer_relevance"), REL_GOOD,   REL_OK,    pct=True),
            _abst_fmt(r.get("abstention")),
        )

    t1.columns[2].footer = _conf_fmt(_avg(cols["confidence"]))
    t1.columns[3].footer = _fmt(_avg(cols["accuracy"]),         ACC_GOOD,   ACC_OK)
    t1.columns[4].footer = _fmt(_avg(cols["faithfulness"]),     FAITH_GOOD, FAITH_OK,  pct=True)
    t1.columns[5].footer = _fmt(_avg(cols["answer_relevance"]), REL_GOOD,   REL_OK,    pct=True)

    # ── Table 2: retrieval metrics + latency ──────────────────────────────────
    t2 = Table(
        title=f"[bold cyan]{author}[/bold cyan] — Retrieval & Latency",
        box=box.ROUNDED, show_footer=True, expand=True,
    )
    t2.add_column("#",         width=3,  no_wrap=True)
    t2.add_column("Question",  ratio=3,  overflow="fold")
    t2.add_column("Precision", width=11, justify="right", footer="Avg")
    t2.add_column("Recall",    width=10, justify="right", footer="Avg")
    t2.add_column("TTFT",      width=7,  justify="right")
    t2.add_column("Total",     width=7,  justify="right")
    t2.add_column("Cost $",    width=10, justify="right", footer="Total")

    precs: list[float] = []
    recs:  list[float] = []
    total_cost = 0.0
    for i, r in enumerate(results, 1):
        if "error" in r:
            t2.add_row(str(i), r["question"][:80], "—", "—", "—", "—", "—")
            continue
        p = r.get("context_precision")
        rc = r.get("context_recall")
        cost = r.get("cost", 0.0)
        total_cost += cost
        if p is not None:  precs.append(p)
        if rc is not None: recs.append(rc)
        t2.add_row(
            str(i), r["question"][:80],
            _fmt(p,  PREC_GOOD, PREC_OK, pct=True),
            _fmt(rc, REC_GOOD,  REC_OK,  pct=True),
            f"{r['ttft']:.1f}s",
            f"{r['total']:.1f}s",
            f"${cost:.5f}",
        )

    t2.columns[2].footer = _fmt(_avg(precs), PREC_GOOD, PREC_OK, pct=True)
    t2.columns[3].footer = _fmt(_avg(recs),  REC_GOOD,  REC_OK,  pct=True)
    t2.columns[6].footer = f"[bold]${total_cost:.4f}[/bold]"

    console.print()
    console.print(t1)
    console.print()
    console.print(t2)
    console.print()

    _print_callouts(results)


def _print_callouts(results: list[dict]) -> None:
    ok = [r for r in results if "error" not in r]

    low_conf = [r for r in ok if (r.get("confidence") or 1.0) < CONF_WARN]
    if low_conf:
        console.print(f"[bold yellow]⚠  {len(low_conf)} low-confidence answer(s) — readers should verify sources:[/bold yellow]")
        for r in low_conf:
            console.print(f"   • {r['question'][:90]}")
        console.print()

    halluc = [r for r in ok if (r.get("faithfulness") or 1.0) < FAITH_OK]
    if halluc:
        console.print(f"[bold red]✗  {len(halluc)} answer(s) with faithfulness < 50% — possible hallucination:[/bold red]")
        for r in halluc:
            console.print(f"   • {r['question'][:90]}")
        console.print()

    # The most dangerous case: low confidence but Claude answered confidently anyway
    dangerous = [r for r in ok if r.get("abstention") == 0.0]
    if dangerous:
        console.print(f"[bold red]🚨 {len(dangerous)} question(s) where confidence was low but Claude did NOT hedge:[/bold red]")
        for r in dangerous:
            conf_pct = (r.get("confidence") or 0) * 100
            console.print(f"   • [{conf_pct:.0f}% conf] {r['question'][:85]}")
        console.print()

    irrel = [r for r in ok if (r.get("answer_relevance") or 1.0) < REL_OK]
    if irrel:
        console.print(f"[bold yellow]~  {len(irrel)} answer(s) with relevance < 60% — evasive or off-topic:[/bold yellow]")
        for r in irrel:
            console.print(f"   • {r['question'][:90]}")
        console.print()


# ── Latency report ────────────────────────────────────────────────────────────

def print_latency_report(results: list[dict]) -> None:
    all_ttft:  list[float] = []
    all_total: list[float] = []
    for r in results:
        if "error" in r:
            continue
        all_ttft.extend(r.get("ttft_samples",  [r["ttft"]]))
        all_total.extend(r.get("total_samples", [r["total"]]))

    if not all_ttft:
        return

    n = len(all_ttft)
    table = Table(
        title=f"Latency Distribution  (n={n} samples)",
        box=box.SIMPLE, expand=False,
    )
    table.add_column("",      width=8)
    table.add_column("min",   width=8, justify="right")
    table.add_column("p50",   width=8, justify="right")
    table.add_column("p95",   width=8, justify="right")
    table.add_column("p99",   width=8, justify="right")
    table.add_column("max",   width=8, justify="right")

    for label, data in [("TTFT", all_ttft), ("Total", all_total)]:
        table.add_row(
            label,
            f"{min(data):.2f}s",
            f"{_percentile(data, 50):.2f}s",
            f"{_percentile(data, 95):.2f}s",
            f"{_percentile(data, 99):.2f}s",
            f"{max(data):.2f}s",
        )

    console.print(table)
    console.print()


# ── Regression comparison ─────────────────────────────────────────────────────

def print_regression(results: list[dict], baseline_path: Path) -> None:
    try:
        baseline = json.loads(baseline_path.read_text())
    except Exception as e:
        console.print(f"[red]Cannot load baseline {baseline_path}: {e}[/red]")
        return

    base_by_q = {r["question"]: r for r in baseline if "question" in r}

    metrics = [
        ("confidence",       "Conf%",      True),
        ("accuracy",         "Accuracy",   False),
        ("faithfulness",     "Faithful",   True),
        ("context_precision","Precision",  True),
        ("context_recall",   "Recall",     True),
        ("answer_relevance", "Relevance",  True),
    ]

    table = Table(title="Regression vs Baseline", box=box.ROUNDED, expand=True)
    table.add_column("Question", ratio=3, overflow="fold")
    for _, label, _ in metrics:
        table.add_column(label, width=14, justify="right")

    any_regression = False
    for r in results:
        if "error" in r:
            continue
        b = base_by_q.get(r["question"])
        row = [r["question"][:70]]
        for key, _, pct in metrics:
            cur  = r.get(key)
            prev = b.get(key) if b else None
            if cur is None:
                row.append("[dim]n/a[/dim]")
                continue
            label_cur = f"{cur*100:.0f}%" if pct else f"{cur:.2f}"
            if prev is None:
                row.append(label_cur)
                continue
            delta = cur - prev
            sign  = "+" if delta >= 0 else ""
            if delta > 0.02:
                color = "green"
            elif delta < -0.02:
                color = "red"
                any_regression = True
            else:
                color = "dim"
            row.append(f"[{color}]{label_cur} ({sign}{delta*100:.0f}%)[/{color}]")
        table.add_row(*row)

    console.print()
    console.print(table)
    if any_regression:
        console.print("[bold red]  ↓ Regression detected vs baseline.[/bold red]")
    else:
        console.print("[green]  ✓ No regression vs baseline.[/green]")
    console.print()


# ── Save ──────────────────────────────────────────────────────────────────────

def save_report(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"[dim]Report saved → {path}[/dim]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pulsebot evals against a live server.")
    parser.add_argument("--author",     required=True, help="Author handle to eval")
    parser.add_argument("--server",     default="http://localhost:8000")
    parser.add_argument("--golden",     help="Path to golden.json  (default: evals/<author>/golden.json)")
    parser.add_argument("--out",        help="Path for JSON report  (default: evals/<author>/report.json)")
    parser.add_argument("--runs",       type=int, default=1,
                        help="Repeat each question N times for latency percentiles (default: 1)")
    parser.add_argument("--baseline",   help="Previous report.json to compare against for regression")
    parser.add_argument("--llm-judge",  action="store_true",
                        help="Use Claude Haiku as accuracy judge instead of cosine similarity")
    parser.add_argument("--skip-llm",   action="store_true",
                        help="Skip all LLM-based scorers (faithfulness, precision, recall, relevance)")
    args = parser.parse_args()

    golden_path = Path(args.golden) if args.golden else Path(f"evals/{args.author}/golden.json")
    if not golden_path.exists():
        console.print(f"[red]Golden set not found: {golden_path}[/red]")
        console.print("[dim]Each entry needs 'question'; optionally 'expected_answer' or 'expected_keywords'.[/dim]")
        sys.exit(1)

    golden = json.loads(golden_path.read_text())
    if not isinstance(golden, list) or not golden:
        console.print("[red]golden.json must be a non-empty JSON array.[/red]")
        sys.exit(1)

    mode_flags = []
    if args.runs > 1:      mode_flags.append(f"runs={args.runs}")
    if args.llm_judge:     mode_flags.append("llm-judge")
    if args.skip_llm:      mode_flags.append("skip-llm")
    mode_str = "  " + "  ".join(mode_flags) if mode_flags else ""

    console.print(
        f"\n[bold cyan]Pulsebot Evals[/bold cyan]"
        f"  author=[cyan]{args.author}[/cyan]"
        f"  questions={len(golden)}"
        f"{mode_str}\n"
    )

    results = run_eval(
        server=args.server,
        author=args.author,
        golden=golden,
        use_llm_judge=args.llm_judge,
        skip_llm=args.skip_llm,
        n_runs=args.runs,
    )

    print_report(args.author, results)
    print_latency_report(results)

    if args.baseline:
        print_regression(results, Path(args.baseline))

    out_path = Path(args.out) if args.out else Path(f"evals/{args.author}/report.json")
    save_report(results, out_path)


if __name__ == "__main__":
    main()
