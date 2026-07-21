#!/usr/bin/env python3
"""Run LAK24 analytics and write mockup_data.js for mockup.html."""

import json
import random
from pathlib import Path

import numpy as np

import reasoning_disagreement_lak24 as rd
import similar_pairs_lak24 as sp
from consensus_coding import cosine, embed, extract_and_complete_code, justification_of

SAMPLE_SIZE_START = 20
SAMPLE_SIZE_MAX = 60
PANEL1_TOP = 8
PANEL2_TOP = 8
OUT_PATH = Path(__file__).resolve().parent / "mockup_data.js"

random.seed(0)


def build_panel2_pairs() -> list[dict]:
    rows = sp.load_rows(sp.CSV_PATH)
    vecs = np.array(embed([r["text"] for r in rows]), dtype=float)
    unit = sp.normalize_rows(vecs)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -1.0)

    n = len(rows)
    bucket1_directed: list[tuple[int, int, float]] = []
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
        if sp.pair_bucket(rows[i]["codes"], rows[best_j]["codes"]) == 1:
            bucket1_directed.append((i, best_j, best_s))

    bucket1 = sp.dedupe_sort(bucket1_directed)
    pairs: list[dict] = []
    for i, j, s in bucket1[:PANEL2_TOP]:
        pairs.append(
            {
                "itemA": {
                    "text": rows[i]["text"],
                    "human_code": rows[i]["label"],
                },
                "itemB": {
                    "text": rows[j]["text"],
                    "human_code": rows[j]["label"],
                },
                "similarity": round(s, 3),
            }
        )
    return pairs


def _codes_label(active: list[str]) -> str:
    return ", ".join(active) if active else "(none)"


def build_panel1_items() -> list[dict]:
    rows = sp.load_rows(sp.CSV_PATH)
    random.seed(0)
    order = list(range(len(rows)))
    random.shuffle(order)

    genuine: list[dict] = []
    target = SAMPLE_SIZE_START
    n_processed = 0

    for idx in order:
        if len(genuine) >= PANEL1_TOP:
            break
        if n_processed >= SAMPLE_SIZE_MAX:
            break

        row = rows[idx]
        text = row["text"]
        n_processed += 1
        print(f"  Panel 1 LLM [{n_processed}/{target}] …")
        msg_a = rd.code_utterance("Agent A", "bold and decisive", text)
        msg_b = rd.code_utterance("Agent B", "cautious and conservative", text)
        codes_a = extract_and_complete_code(msg_a, rd.SRL_KEYS)
        codes_b = extract_and_complete_code(msg_b, rd.SRL_KEYS)
        active_a = sorted(k for k, v in codes_a.items() if v == 1)
        active_b = sorted(k for k, v in codes_b.items() if v == 1)
        just_a = justification_of(msg_a)
        just_b = justification_of(msg_b)
        u, v = embed([just_a, just_b])
        sim = float(cosine(u, v))

        if active_a != active_b:
            genuine.append(
                {
                    "text": text,
                    "human_code": row["label"],
                    "reasoningA": just_a,
                    "reasoningB": just_b,
                    "similarity": round(sim, 3),
                    "codesA": _codes_label(active_a),
                    "codesB": _codes_label(active_b),
                }
            )

        if (
            n_processed >= target
            and len(genuine) < PANEL1_TOP
            and target < SAMPLE_SIZE_MAX
        ):
            target = min(target + 10, SAMPLE_SIZE_MAX)
            print(
                f"  … expanding sample to {target} utterances "
                f"(genuine label disagreements so far: {len(genuine)})"
            )

    genuine.sort(key=lambda x: x["similarity"])
    return genuine[:PANEL1_TOP]


def main() -> None:
    print("Building Panel 2 (similar pairs, bucket 1)…")
    panel2 = build_panel2_pairs()
    print(f"  {len(panel2)} pairs")

    print("Building Panel 1 (genuine label disagreements, two agents)…")
    panel1 = build_panel1_items()
    print(f"  {len(panel1)} genuine disagreement items (target {PANEL1_TOP})")

    js = (
        f"window.PANEL1_DATA = {json.dumps(panel1, ensure_ascii=False)};\n"
        f"window.PANEL2_DATA = {json.dumps(panel2, ensure_ascii=False)};\n"
    )
    OUT_PATH.write_text(js, encoding="utf-8")
    print(f"Wrote {OUT_PATH.name}: PANEL1_DATA={len(panel1)} items, PANEL2_DATA={len(panel2)} pairs")


if __name__ == "__main__":
    main()
