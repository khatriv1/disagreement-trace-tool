#!/usr/bin/env python3
"""Cluster codebook revision texts and relate clusters to agreement help/harm."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from embedding_revision_analysis import collect_steps, embed_texts

CONRAD = Path(__file__).resolve().parent
OUT_TXT = CONRAD / "revision_clusters.txt"
OUT_CSV = CONRAD / "revision_clusters.csv"

K_VALUES = [3, 4, 5]
PROTOTYPE_N = 3
TEXT_TRIM = 300


def trim_text(text: str, n: int = TEXT_TRIM) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vecs / norms


def choose_k(vecs: np.ndarray) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    best_k = K_VALUES[0]
    best_score = -1.0
    n = len(vecs)
    for k in K_VALUES:
        if k >= n:
            scores[k] = float("nan")
            continue
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(vecs)
        if len(set(labels)) < 2:
            scores[k] = float("nan")
            continue
        s = float(silhouette_score(vecs, labels))
        scores[k] = s
        if s > best_score:
            best_score = s
            best_k = k
    return best_k, scores


def suggest_label(cluster_items: list[dict], mean_words: float) -> str:
    sample = " ".join(trim_text(it["revision_text"], 800) for it in cluster_items[:5]).lower()
    parts: list[str] = []
    if mean_words > 15:
        parts.append("adds wording")
    elif mean_words < -5:
        parts.append("trims/shortens definitions")
    else:
        parts.append("rewrites without large net word change")

    if "include" in sample or "exclude" in sample or "not " in sample:
        parts.append("include/exclude rules")
    if "example" in sample or "e.g." in sample or "such as" in sample:
        parts.append("adds examples")
    if "do not" in sample or "does not" in sample or "not include" in sample:
        parts.append("boundary clarifications")
    if "must" in sample or "should" in sample or "only" in sample:
        parts.append("tightens criteria")

    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts[:3])


def prototypical(items: list[dict], unit: np.ndarray, labels: np.ndarray, cid: int) -> list[dict]:
    idx = [i for i, lab in enumerate(labels) if lab == cid]
    if not idx:
        return []
    sub = unit[idx]
    centroid = sub.mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn > 0:
        centroid = centroid / cn
    sims = sub @ centroid
    order = sorted(range(len(idx)), key=lambda j: sims[j], reverse=True)
    return [items[idx[j]] for j in order[:PROTOTYPE_N]]


def main() -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("REVISION CLUSTER ANALYSIS")
    lines.append("Model: all-MiniLM-L6-v2 (sentence-transformers)")
    lines.append("=" * 72)
    lines.append("")

    steps, used_runs, skipped_runs = collect_steps()
    lines.append("RUNS USED:")
    for u in used_runs:
        lines.append(f"  + {u}")
    lines.append("")
    lines.append("RUNS SKIPPED:")
    for s in skipped_runs:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append(f"Total revision steps: {len(steps)}")
    lines.append("")

    if len(steps) < 4:
        raise SystemExit("Need at least 4 revision steps for clustering.")

    print(f"Embedding {len(steps)} revision texts …", flush=True)
    vecs = embed_texts([s["revision_text"] for s in steps])
    unit = normalize_rows(np.asarray(vecs, dtype=float))

    chosen_k, sil_scores = choose_k(unit)
    lines.append("SILHOUETTE SCORES (higher = better-separated clusters):")
    for k in K_VALUES:
        sc = sil_scores[k]
        mark = "  <-- chosen" if k == chosen_k else ""
        lines.append(
            f"  k={k}: {sc:.4f}{mark}" if not np.isnan(sc) else f"  k={k}: n/a{mark}"
        )
    lines.append(f"Using k={chosen_k} for cluster report.")
    lines.append("")

    labels = KMeans(n_clusters=chosen_k, random_state=0, n_init=10).fit_predict(unit)

    for i, step in enumerate(steps):
        step["cluster_id"] = int(labels[i])

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run", "round", "delta_kappa", "helped", "n_words_added", "cluster_id"],
        )
        w.writeheader()
        for s in steps:
            w.writerow(
                {
                    "run": s["run"],
                    "round": s["round"],
                    "delta_kappa": f"{s['delta_kappa']:.6f}",
                    "helped": s["helped"],
                    "n_words_added": s["n_words_added"],
                    "cluster_id": s["cluster_id"],
                }
            )

    cluster_summaries: list[dict] = []
    for cid in range(chosen_k):
        members = [s for s in steps if s["cluster_id"] == cid]
        deltas = [s["delta_kappa"] for s in members]
        helped = sum(s["helped"] for s in members)
        hurt = len(members) - helped
        mean_words = float(np.mean([s["n_words_added"] for s in members]))
        label = suggest_label(members, mean_words)
        protos = prototypical(steps, unit, labels, cid)

        lines.append("=" * 72)
        lines.append(f"CLUSTER {cid} — suggested label: {label}")
        lines.append("=" * 72)
        lines.append(f"  size: {len(members)} revisions")
        lines.append(f"  mean delta_kappa: {np.mean(deltas):+.4f}")
        lines.append(f"  helped vs hurt: {helped} helped, {hurt} hurt")
        lines.append(f"  mean n_words_added: {mean_words:+.1f}")
        lines.append("  prototypical revisions (closest to centroid):")
        for p in protos:
            lines.append(
                f"    - {p['run']} r{p['round']}  delta_kappa={p['delta_kappa']:+.4f}  "
                f"words_added={p['n_words_added']:+d}"
            )
            lines.append(f"      {trim_text(p['revision_text'])}")
        lines.append("")

        cluster_summaries.append(
            {
                "id": cid,
                "label": label,
                "size": len(members),
                "mean_delta": float(np.mean(deltas)),
                "helped": helped,
                "hurt": hurt,
            }
        )

    lines.append("=" * 72)
    lines.append("PLAIN-LANGUAGE SUMMARY")
    lines.append("=" * 72)
    for c in sorted(cluster_summaries, key=lambda x: x["mean_delta"], reverse=True):
        tendency = "tended to help" if c["mean_delta"] > 0 else "tended to hurt"
        if abs(c["mean_delta"]) < 0.005:
            tendency = "had mixed/neutral effect on"
        lines.append(
            f"  Cluster {c['id']} ({c['label']}): {c['size']} revisions, "
            f"mean delta_kappa {c['mean_delta']:+.4f}, {c['helped']} helped / {c['hurt']} hurt — "
            f"{tendency} agreement."
        )

    lines.append("")
    lines.append(f"Saved: {OUT_CSV.name}")
    lines.append(f"Saved: {OUT_TXT.name}")

    report = "\n".join(lines) + "\n"
    OUT_TXT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
