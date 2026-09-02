#!/usr/bin/env python3
"""
part3_significance.py
Bootstrap + permutation tests on the completed weak-start run.
Reuses scored caches from part3_weakstart_results/ — no LLM calls.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
IN_DIR = BASE / "part3_weakstart_results"
OUT_TXT = BASE / "part3_significance_results.txt"

TUT_CODES = [
    "Greeting",
    "Instruction",
    "Guiding feedback",
    "Aligning to prior knowledge",
    "Understanding/Engagement-Tutor",
    "Encouragement",
]
TUT_FILES = [
    "Data_2/First Author Copy GPT-Then-Human - Transcript B.csv",
    "Data_2/First Author Copy GPT-Then-Human - transcript C.csv",
]
FOCUS = "Understanding/Engagement-Tutor"

N_BOOT = 2000
N_PERM = 2000
SEED = 42


def parse01(value) -> int:
    s = str(value or "").strip().casefold()
    return 1 if s in ("1", "1.0", "yes", "true") else 0


def human_label_for_code(header: list[str], values: list[str], code: str) -> int:
    target = code.strip().casefold()
    matched = [
        parse01(v)
        for n, v in zip(header, values)
        if (n or "").strip().casefold() == target
    ]
    return max(matched) if matched else 0


def load_human_gold(base: Path) -> dict[str, dict[str, int]]:
    """item_id -> {code: 0/1} in utterance order."""
    gold: dict[str, dict[str, int]] = {}
    idx = 0
    for rel in TUT_FILES:
        path = base / rel
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            si = ti = None
            for i, name in enumerate(header):
                n = (name or "").strip().casefold()
                if si is None and n == "speaker_type":
                    si = i
                if ti is None and n == "text":
                    ti = i
            for values in reader:
                if len(values) < len(header):
                    values += [""] * (len(header) - len(values))
                if (values[si] or "").strip().casefold() != "tutor":
                    continue
                if not (values[ti] or "").strip():
                    continue
                idx += 1
                iid = f"tuto_{idx:04d}"
                gold[iid] = {
                    c: human_label_for_code(header, values, c) for c in TUT_CODES
                }
    return gold


def load_scored(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["item_id"]] = {c: int(row.get(c, 0) or 0) for c in TUT_CODES}
    return out


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    p_pred_pos = (tp + fp) / n
    p_pred_neg = (tn + fn) / n
    p_gold_pos = (tp + fn) / n
    p_gold_neg = (tn + fp) / n
    pe = p_pred_pos * p_gold_pos + p_pred_neg * p_gold_neg
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def kappa_from_vecs(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    tn = int(np.sum((pred == 0) & (gold == 0)))
    return cohen_kappa(tp, fp, fn, tn)


def build_arrays(
    item_ids: list[str],
    gold_map: dict[str, dict[str, int]],
    pred0: dict[str, dict[str, int]],
    pred_r: dict[str, dict[str, int]],
    codes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Return flat pred0, pred_r, gold arrays and utterance index per decision.
    One row per (utterance, code) in item_ids order.
    """
    codes = codes or TUT_CODES
    p0, pr, g, utt_idx = [], [], [], []
    for ui, iid in enumerate(item_ids):
        for code in codes:
            p0.append(pred0[iid][code])
            pr.append(pred_r[iid][code])
            g.append(gold_map[iid][code])
            utt_idx.append(ui)
    return (
        np.array(p0, dtype=int),
        np.array(pr, dtype=int),
        np.array(g, dtype=int),
        utt_idx,
    )


def pooled_kappa_by_utterances(
    item_ids: list[str],
    subset_ids: list[str] | None,
    gold_map: dict[str, dict[str, int]],
    pred_map: dict[str, dict[str, int]],
    codes: list[str] | None = None,
) -> float:
    ids = subset_ids if subset_ids is not None else item_ids
    p0, _, g, _ = build_arrays(ids, gold_map, pred_map, pred_map, codes)
    return kappa_from_vecs(p0, g)


def bootstrap_kappa_diff(
    item_ids: list[str],
    gold_map: dict[str, dict[str, int]],
    pred0: dict[str, dict[str, int]],
    pred_r: dict[str, dict[str, int]],
    codes: list[str] | None = None,
    n_resamples: int = N_BOOT,
    rng: random.Random | None = None,
) -> dict:
    rng = rng or random.Random(SEED)
    codes = codes or TUT_CODES
    n = len(item_ids)

    obs_p0, obs_pr, obs_g, utt_idx = build_arrays(item_ids, gold_map, pred0, pred_r, codes)
    obs_diff = kappa_from_vecs(obs_pr, obs_g) - kappa_from_vecs(obs_p0, obs_g)

    diffs: list[float] = []
    for _ in range(n_resamples):
        sampled = [rng.randrange(n) for _ in range(n)]
        # map utterance index -> included in resample (with multiplicity)
        boot_p0, boot_pr, boot_g = [], [], []
        for ui in sampled:
            iid = item_ids[ui]
            for code in codes:
                boot_p0.append(pred0[iid][code])
                boot_pr.append(pred_r[iid][code])
                boot_g.append(gold_map[iid][code])
        bp0 = np.array(boot_p0, dtype=int)
        bpr = np.array(boot_pr, dtype=int)
        bg = np.array(boot_g, dtype=int)
        diffs.append(kappa_from_vecs(bpr, bg) - kappa_from_vecs(bp0, bg))

    diffs_arr = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs_arr, [2.5, 97.5])
    share_ge0 = float(np.mean(diffs_arr >= 0))

    return {
        "obs_diff": obs_diff,
        "kappa_baseline": kappa_from_vecs(obs_p0, obs_g),
        "kappa_revised": kappa_from_vecs(obs_pr, obs_g),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "share_ge0": share_ge0,
    }


def permutation_kappa_diff(
    item_ids: list[str],
    gold_map: dict[str, dict[str, int]],
    pred0: dict[str, dict[str, int]],
    pred_r: dict[str, dict[str, int]],
    codes: list[str] | None = None,
    n_shuffles: int = N_PERM,
    rng: random.Random | None = None,
) -> dict:
    rng = rng or random.Random(SEED + 1)
    codes = codes or TUT_CODES

    p0, pr, g, _ = build_arrays(item_ids, gold_map, pred0, pred_r, codes)
    obs_diff = kappa_from_vecs(pr, g) - kappa_from_vecs(p0, g)

    perm_diffs: list[float] = []
    for _ in range(n_shuffles):
        pp0 = p0.copy()
        ppr = pr.copy()
        for i in range(len(p0)):
            if rng.random() < 0.5:
                pp0[i], ppr[i] = ppr[i], pp0[i]
        perm_diffs.append(kappa_from_vecs(ppr, g) - kappa_from_vecs(pp0, g))

    perm_arr = np.abs(np.array(perm_diffs))
    p_two = float(np.mean(perm_arr >= abs(obs_diff)))

    return {"obs_diff": obs_diff, "p_two_sided": p_two}


def verdict(diff: float, ci_lo: float, ci_hi: float, p_val: float) -> str:
    if ci_hi < 0:
        return "revision reliably REDUCES agreement with human gold"
    if ci_lo > 0:
        return "revision reliably IMPROVES agreement with human gold"
    if p_val < 0.05 and diff < 0:
        return "revision significantly reduces agreement (permutation p<0.05)"
    if p_val < 0.05 and diff > 0:
        return "revision significantly improves agreement (permutation p<0.05)"
    return "no reliable difference between baseline and revised codebook"


def best_revision_round(results_csv: Path) -> tuple[int, float]:
    best_r, best_k = 0, -1.0
    with results_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = int(row["round"])
            if r == 0:
                continue
            k = float(row["pooled_kappa"])
            if k > best_k:
                best_k, best_r = k, r
    return best_r, best_k


def run_comparison(
    label: str,
    round_num: int,
    item_ids: list[str],
    gold_map: dict[str, dict[str, int]],
    pred0: dict[str, dict[str, int]],
    pred_r: dict[str, dict[str, int]],
    lines: list[str],
    rng: random.Random,
) -> None:
    lines.append("=" * 72)
    lines.append(label)
    lines.append(f"Comparison: round 0 (baseline) vs round {round_num} (revised)")
    lines.append("=" * 72)

    # ── Pooled (all codes) ────────────────────────────────────────────────
    boot = bootstrap_kappa_diff(item_ids, gold_map, pred0, pred_r, TUT_CODES, N_BOOT, rng)
    perm = permutation_kappa_diff(item_ids, gold_map, pred0, pred_r, TUT_CODES, N_PERM, rng)

    lines.append("")
    lines.append("POOLED (all code-decisions):")
    lines.append(
        f"  Baseline kappa (round 0):  {boot['kappa_baseline']:.4f}"
    )
    lines.append(
        f"  Revised kappa (round {round_num}): {boot['kappa_revised']:.4f}"
    )
    lines.append(
        f"  Observed difference (revised - baseline): {boot['obs_diff']:+.4f}"
    )
    lines.append(
        f"  Bootstrap 95% CI of difference: [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]"
    )
    lines.append(
        f"  Share of bootstrap resamples with diff >= 0: {boot['share_ge0']:.3f}"
    )
    lines.append(f"  Permutation p-value (two-sided): {perm['p_two_sided']:.4f}")
    v = verdict(boot["obs_diff"], boot["ci_lo"], boot["ci_hi"], perm["p_two_sided"])
    lines.append(f"  VERDICT: {v}")

    print(f"\n{label}")
    print(f"  Pooled: diff={boot['obs_diff']:+.4f}  "
          f"95% CI=[{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]  "
          f"p={perm['p_two_sided']:.4f}")
    print(f"  -> {v}")

    # ── UE-Tutor alone (bootstrap only) ───────────────────────────────────
    boot_ue = bootstrap_kappa_diff(
        item_ids, gold_map, pred0, pred_r, [FOCUS], N_BOOT, rng
    )
    lines.append("")
    lines.append(f"{FOCUS} (bootstrap only):")
    lines.append(f"  Baseline kappa: {boot_ue['kappa_baseline']:.4f}")
    lines.append(f"  Revised kappa:  {boot_ue['kappa_revised']:.4f}")
    lines.append(f"  Observed difference: {boot_ue['obs_diff']:+.4f}")
    lines.append(
        f"  Bootstrap 95% CI: [{boot_ue['ci_lo']:+.4f}, {boot_ue['ci_hi']:+.4f}]"
    )
    lines.append(f"  Share bootstrap diff >= 0: {boot_ue['share_ge0']:.3f}")

    if boot_ue["ci_hi"] < 0:
        ue_v = "UE revision reliably reduces agreement"
    elif boot_ue["ci_lo"] > 0:
        ue_v = "UE revision reliably improves agreement"
    else:
        ue_v = "no reliable UE kappa difference (bootstrap CI spans zero)"
    lines.append(f"  VERDICT: {ue_v}")

    print(f"  UE-Tutor: diff={boot_ue['obs_diff']:+.4f}  "
          f"95% CI=[{boot_ue['ci_lo']:+.4f}, {boot_ue['ci_hi']:+.4f}]  "
          f"-> {ue_v}")
    lines.append("")


def main() -> None:
    print("=" * 72)
    print("PART 3 SIGNIFICANCE — weak-start baseline vs revised (no LLM re-runs)")
    print("=" * 72)

    gold_map = load_human_gold(BASE)
    pred0 = load_scored(IN_DIR / "scored_round0_baseline.csv")
    item_ids = sorted(pred0.keys(), key=lambda x: int(x.split("_")[1]))

    if len(item_ids) != len(gold_map):
        print(
            f"WARNING: scored rows ({len(item_ids)}) != gold rows ({len(gold_map)})"
        )

    best_r, best_k = best_revision_round(IN_DIR / "part3_weakstart_results.csv")
    print(f"\nLoaded {len(item_ids)} utterances")
    print(f"Best post-revision round: {best_r} (pooled kappa={best_k:.4f})")

    preds: dict[int, dict[str, dict[str, int]]] = {0: pred0}
    for r in range(1, 6):
        path = IN_DIR / (f"scored_round{r}.csv")
        preds[r] = load_scored(path)

    lines: list[str] = [
        "Part 3 Significance Test — Weak-Start Run",
        f"Bootstrap resamples: {N_BOOT} | Permutation shuffles: {N_PERM}",
        f"N utterances: {len(item_ids)} | N pooled decisions: {len(item_ids) * len(TUT_CODES)}",
        f"Best post-revision round: {best_r} (pooled kappa={best_k:.4f})",
        "",
    ]

    rng = random.Random(SEED)

    run_comparison(
        "(i) Round 0 vs Round 5 (baseline vs final)",
        5,
        item_ids,
        gold_map,
        pred0,
        preds[5],
        lines,
        rng,
    )

    run_comparison(
        f"(ii) Round 0 vs Round {best_r} (baseline vs best post-revision)",
        best_r,
        item_ids,
        gold_map,
        pred0,
        preds[best_r],
        lines,
        rng,
    )

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {OUT_TXT}")


if __name__ == "__main__":
    main()
