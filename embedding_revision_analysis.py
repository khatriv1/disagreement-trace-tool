#!/usr/bin/env python3
"""Predict revision help/harm from revision-text embeddings (no model re-runs)."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CONRAD = Path(__file__).resolve().parent
PKG = CONRAD / "part3_experiment_package"
DATA_MORE = PKG / "results"
OUT_TXT = CONRAD / "embedding_revision_analysis.txt"
OUT_CSV = CONRAD / "revision_steps.csv"

# Candidate runs: (run_id, folder). Missing folders are skipped.
CANDIDATE_RUNS: list[tuple[str, Path]] = [
    ("label", DATA_MORE / "label"),
    ("reasoning", DATA_MORE / "reasoning"),
    ("ambiguity", DATA_MORE / "ambiguity"),
    ("all_three", PKG / "results/all_three"),
    ("label_ambiguity", PKG / "results/label_ambiguity"),
    ("label_reasoning", PKG / "results/label_reasoning"),
    ("reasoning_ambiguity", PKG / "results/reasoning_ambiguity"),
    ("bare_all_three_seed42", PKG / "results_bare_all_three_seed42"),
    ("bare_all_three_seed123", PKG / "results_bare_all_three_seed123"),
    ("bare_all_three_seed7", PKG / "results_bare_all_three_seed7"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))


def codebook_codes(cb: dict) -> dict[str, str]:
    return {k: norm(v) for k, v in cb.get("codes", {}).items()}


def revision_text(prev_cb: dict, curr_cb: dict) -> tuple[str, int, list[str]]:
    prev = codebook_codes(prev_cb)
    curr = codebook_codes(curr_cb)
    changed: list[str] = []
    for code in sorted(set(prev) | set(curr)):
        old, new = prev.get(code, ""), curr.get(code, "")
        if old != new:
            changed.append(new)
    text = " ".join(changed).strip()
    return text, len(changed), changed


def load_kappa_by_round(run_dir: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = int(row["round"])
            rows[r] = row
    return rows


def collect_steps() -> tuple[list[dict], list[str], list[str]]:
    used_runs: list[str] = []
    skipped_runs: list[str] = []
    steps: list[dict] = []

    for run_id, run_dir in CANDIDATE_RUNS:
        if not run_dir.is_dir() or not (run_dir / "results.csv").exists():
            skipped_runs.append(f"{run_id} ({run_dir})")
            continue
        if not (run_dir / "codebook_round0.json").exists():
            skipped_runs.append(f"{run_id} (missing codebook_round0.json)")
            continue

        kappa_rows = load_kappa_by_round(run_dir)
        run_steps = 0
        for r in range(1, 6):
            rev_path = run_dir / f"revised_codebook_round{r}.json"
            if not rev_path.exists() or r not in kappa_rows or (r - 1) not in kappa_rows:
                continue
            prev_path = (
                run_dir / "codebook_round0.json"
                if r == 1
                else run_dir / f"revised_codebook_round{r - 1}.json"
            )
            if not prev_path.exists():
                continue

            prev_cb = load_json(prev_path)
            curr_cb = load_json(rev_path)
            text, n_codes_changed, _ = revision_text(prev_cb, curr_cb)
            if not text:
                continue

            row_r = kappa_rows[r]
            row_prev = kappa_rows[r - 1]
            delta = float(row_r["pooled_kappa"]) - float(row_prev["pooled_kappa"])
            n_words_added = int(row_r["codebook_word_count"]) - int(row_prev["codebook_word_count"])

            steps.append(
                {
                    "run": run_id,
                    "round": r,
                    "delta_kappa": delta,
                    "helped": int(delta > 0),
                    "n_words_added": n_words_added,
                    "n_codes_changed": n_codes_changed,
                    "revision_text": text,
                    "n_changes_csv": int(row_r.get("n_changes", n_codes_changed)),
                }
            )
            run_steps += 1

        if run_steps:
            used_runs.append(f"{run_id} -> {run_dir}")
        else:
            skipped_runs.append(f"{run_id} (no revision steps with text changes)")

    return steps, used_runs, skipped_runs


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=False))


def loo_cv_regression(X: np.ndarray, y: np.ndarray, runs: np.ndarray) -> tuple[np.ndarray, float]:
    preds = np.zeros(len(y))
    unique_runs = sorted(set(runs))
    for held in unique_runs:
        test = runs == held
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        pipe.fit(X[train], y[train])
        preds[test] = pipe.predict(X[test])
    r2 = r2_score(y, preds)
    return preds, r2


def loo_cv_classification(
    X: np.ndarray, y: np.ndarray, runs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    preds = np.zeros(len(y), dtype=int)
    probs = np.full(len(y), np.nan)
    unique_runs = sorted(set(runs))
    for held in unique_runs:
        test = runs == held
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue
        y_train = y[train]
        if len(set(y_train)) < 2:
            maj = int(round(y_train.mean()))
            preds[test] = maj
            probs[test] = maj
            continue
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        clf.fit(X[train], y_train)
        preds[test] = clf.predict(X[test])
        probs[test] = clf.predict_proba(X[test])[:, 1]
    acc = accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = None
    return preds, probs, acc, auc


def corr_report(x: np.ndarray, y: np.ndarray, name: str, lines: list[str]) -> None:
    if len(x) < 3 or np.std(x) == 0:
        lines.append(f"  {name}: insufficient variation")
        return
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    lines.append(
        f"  {name}: Pearson r={pr.statistic:.4f} (p={pr.pvalue:.4f}), "
        f"Spearman rho={sr.statistic:.4f} (p={sr.pvalue:.4f})"
    )


def main() -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EMBEDDING REVISION ANALYSIS")
    lines.append("Model: all-MiniLM-L6-v2 (sentence-transformers)")
    lines.append("=" * 72)
    lines.append("")

    steps, used_runs, skipped_runs = collect_steps()
    lines.append("RUNS USED:")
    for u in used_runs:
        lines.append(f"  + {u}")
    lines.append("")
    lines.append("RUNS SKIPPED:")
    if skipped_runs:
        for s in skipped_runs:
            lines.append(f"  - {s}")
    else:
        lines.append("  (none)")
    lines.append("")

    if not steps:
        lines.append("No revision steps found.")
        OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        sys.exit(1)

    # Save revision_steps.csv (no embedding columns)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run", "round", "delta_kappa", "helped", "n_words_added", "n_codes_changed"],
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
                    "n_codes_changed": s["n_codes_changed"],
                }
            )

    y = np.array([s["delta_kappa"] for s in steps])
    y_bin = np.array([s["helped"] for s in steps])
    runs = np.array([s["run"] for s in steps])
    words = np.array([s["n_words_added"] for s in steps], dtype=float)
    codes = np.array([s["n_codes_changed"] for s in steps], dtype=float)
    texts = [s["revision_text"] for s in steps]

    lines.append(f"Total revision steps (with non-empty text changes): {len(steps)}")
    lines.append(f"Unique runs: {len(set(runs))}")
    lines.append(f"Helped (delta_kappa > 0): {y_bin.sum()} / {len(y_bin)} ({100*y_bin.mean():.1f}%)")
    lines.append("")

    # 1) Sanity baseline correlations
    lines.append("=" * 72)
    lines.append("1) SANITY BASELINE — correlation with delta_kappa")
    lines.append("=" * 72)
    corr_report(words, y, "n_words_added (net codebook word-count change)", lines)
    corr_report(codes, y, "n_codes_changed (definitions that differed)", lines)
    X_plain = np.column_stack([words, codes])
    if np.linalg.matrix_rank(X_plain) >= 1:
        _, r2_plain = loo_cv_regression(X_plain, y, runs)
        _, _, acc_plain, auc_plain = loo_cv_classification(X_plain, y_bin, runs)
        lines.append(f"  Plain-features LOO-CV ridge R² (words+codes): {r2_plain:.4f}")
        lines.append(f"  Plain-features LOO-CV accuracy: {acc_plain:.4f}")
        lines.append(
            f"  Plain-features LOO-CV AUC: {auc_plain:.4f}"
            if auc_plain is not None
            else "  Plain-features LOO-CV AUC: n/a"
        )
    lines.append("")

    # 2) Embedding models
    lines.append("=" * 72)
    lines.append("2) EMBEDDING PREDICTION — leave-one-run-out CV")
    lines.append("=" * 72)
    lines.append("Embedding revision texts ...")
    print("Embedding texts with all-MiniLM-L6-v2 ...", flush=True)
    X_emb = embed_texts(texts)
    lines.append(f"Embedding shape: {X_emb.shape[0]} steps x {X_emb.shape[1]} dims")
    lines.append("")

    _, r2_emb = loo_cv_regression(X_emb, y, runs)
    _, _, acc_emb, auc_emb = loo_cv_classification(X_emb, y_bin, runs)

    maj_class = int(round(y_bin.mean()))
    maj_acc = accuracy_score(y_bin, np.full_like(y_bin, maj_class))

    lines.append(f"Regression (predict delta_kappa):")
    lines.append(f"  LOO-CV R² (embedding ridge): {r2_emb:.4f}")
    lines.append(f"  LOO-CV R² (plain words+codes ridge): {r2_plain:.4f}")
    lines.append("")
    lines.append(f"Classification (helped vs hurt, delta_kappa > 0):")
    lines.append(f"  Majority-class baseline accuracy: {maj_acc:.4f} (predict always {'helped' if maj_class else 'hurt'})")
    lines.append(f"  LOO-CV accuracy (embedding): {acc_emb:.4f}")
    lines.append(f"  LOO-CV accuracy (plain words+codes): {acc_plain:.4f}")
    lines.append(
        f"  LOO-CV AUC (embedding): {auc_emb:.4f}"
        if auc_emb is not None
        else "  LOO-CV AUC (embedding): n/a"
    )
    lines.append(
        f"  LOO-CV AUC (plain words+codes): {auc_plain:.4f}"
        if auc_plain is not None
        else "  LOO-CV AUC (plain words+codes): n/a"
    )
    lines.append("")

    # 3) Sample size caveat + verdict
    lines.append("=" * 72)
    lines.append("3) SAMPLE SIZE & INTERPRETATION")
    lines.append("=" * 72)
    lines.append(
        f"N = {len(steps)} revision steps across {len(set(runs))} runs is small relative to "
        f"embedding dimension ({X_emb.shape[1]}). Expect high variance and risk of overfitting "
        "even with LOO-CV; treat results as exploratory, not confirmatory."
    )
    lines.append("")

    emb_beats_plain_r2 = r2_emb > r2_plain
    emb_beats_plain_acc = acc_emb > acc_plain
    if emb_beats_plain_r2 and emb_beats_plain_acc:
        verdict = (
            "Embedding predictions modestly outperform the plain word-count/code-count baseline on "
            "both regression R² and classification accuracy, but sample size is too small to claim "
            "reliable predictive signal."
        )
    elif emb_beats_plain_r2 or emb_beats_plain_acc:
        verdict = (
            "Embeddings show a slight edge over the plain baseline on one metric but not both; "
            "there is no strong evidence that revision-text embeddings predict help vs harm."
        )
    else:
        verdict = (
            "Revision-text embeddings do NOT predict whether a revision helped agreement better than "
            "the simple word-count / code-count baseline (and both are weak)."
        )

    lines.append("VERDICT:")
    lines.append(f"  {verdict}")
    lines.append("")
    lines.append(f"Saved: {OUT_CSV.name}")
    lines.append(f"Saved: {OUT_TXT.name}")

    report = "\n".join(lines) + "\n"
    OUT_TXT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
