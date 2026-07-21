#!/usr/bin/env python3
"""
Conrad's "most similar item, different human code" analytic on LAK24-coded utterances.
Uses embeddings + human labels only (no LLM agents; Ollama not required).

Process / plan / act code definitions are needed later for LLM agent runs; here we only
use the Yes/No labels. Pairs with high similarity but different codes are candidates where
the codebook is likely ambiguous.
"""

import csv
import re
from pathlib import Path

import numpy as np

from consensus_coding import embed

CSV_PATH = Path(__file__).resolve().parent / "lak24-coded-utterances.csv"
CODE_COLS = ["process", "plan", "act"]
BUCKET1_TOP = 20
BUCKET2_TOP = 10
SNIPPET_LEN = 100


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("remove_flag", "").strip().lower() == "yes":
                continue
            text = row.get("utterance_combined", "").strip()
            if not text:
                continue
            codes = tuple(
                name
                for name in CODE_COLS
                if row.get(name, "").strip().lower() == "yes"
            )
            label = "+".join(codes) if codes else "(none)"
            rows.append({"text": text, "codes": codes, "label": label})
    return rows


def normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vecs / norms


def snippet(text: str, n: int = SNIPPET_LEN) -> str:
    t = text.replace("\n", " ")
    return t if len(t) <= n else t[: n - 3] + "..."


def _normalize_for_dup(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_near_duplicate(text_a: str, text_b: str) -> bool:
    a = _normalize_for_dup(text_a)
    b = _normalize_for_dup(text_b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    words_a = a.split()
    words_b = b.split()
    if not words_a or not words_b:
        return False
    set_a, set_b = set(words_a), set(words_b)
    overlap = len(set_a & set_b) / min(len(set_a), len(set_b))
    return overlap > 0.8


def pair_bucket(codes_i: tuple, codes_j: tuple) -> int | None:
    empty_i = len(codes_i) == 0
    empty_j = len(codes_j) == 0
    if empty_i and empty_j:
        return None
    if empty_i or empty_j:
        return 2
    return 1


def dedupe_sort(pairs: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int, float]] = []
    for i, j, s in pairs:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        unique.append((i, j, s))
    unique.sort(key=lambda t: t[2], reverse=True)
    return unique


def print_pairs(rows: list[dict], pairs: list[tuple[int, int, float]], limit: int) -> None:
    for i, j, s in pairs[:limit]:
        li, lj = rows[i]["label"], rows[j]["label"]
        tag = " [near-dup]" if is_near_duplicate(rows[i]["text"], rows[j]["text"]) else ""
        print(f"{s:.3f}  [{li}] vs [{lj}]{tag}")
        print(f"  A: {snippet(rows[i]['text'])}")
        print(f"  B: {snippet(rows[j]['text'])}")
        print()


def main() -> None:
    rows = load_rows(CSV_PATH)
    print(f"Loaded {len(rows)} coded utterances from {CSV_PATH.name}")

    vecs = np.array(embed([r["text"] for r in rows]), dtype=float)
    unit = normalize_rows(vecs)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -1.0)

    n = len(rows)
    bucket1_directed: list[tuple[int, int, float]] = []
    bucket2_directed: list[tuple[int, int, float]] = []
    for i in range(n):
        best_j = None
        best_s = -2.0
        for j in range(n):
            if i == j or rows[i]["codes"] == rows[j]["codes"]:
                continue
            s = float(sim[i, j])
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is None:
            continue
        b = pair_bucket(rows[i]["codes"], rows[best_j]["codes"])
        if b == 1:
            bucket1_directed.append((i, best_j, best_s))
        elif b == 2:
            bucket2_directed.append((i, best_j, best_s))

    bucket1 = dedupe_sort(bucket1_directed)
    bucket2 = dedupe_sort(bucket2_directed)

    print(
        "\n=== BUCKET 1: both coded, different codes "
        "(strongest codebook-ambiguity signal) ===\n"
    )
    print_pairs(rows, bucket1, BUCKET1_TOP)

    print("=== BUCKET 2: coded vs uncoded ===\n")
    print_pairs(rows, bucket2, BUCKET2_TOP)


if __name__ == "__main__":
    main()
