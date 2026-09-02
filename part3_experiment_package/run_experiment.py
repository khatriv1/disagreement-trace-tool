#!/usr/bin/env python3
"""
run_experiment.py — Part 3 multi-round anchored-holdout codebook revision.

Usage:
  python run_experiment.py --condition label
  python run_experiment.py --condition all_three
  python run_experiment.py --condition all_three --seed 123 --outdir results_all_three_seed123

Conditions select which cases the reviser sees each round (plus model-agreed anchors).
Outputs go to results/<condition>/ with CSV flush after each round.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
DATA = PKG / "data"
RESULTS_ROOT = PKG / "results"
CODEBOOK_PATH = PKG / "codebook_gpt_human.json"

MODEL_A = "qwen2.5:7b"
MODEL_B = "llama3.1:8b"
REVISER_MODEL = "qwen2.5:14b"
TEMP = 0.4
N_ROUNDS = 5
FOCUS = "Understanding/Engagement-Tutor"
ANCHOR_SAMPLE = 5
RANDOM_SEED = 42
ACTIVE_SEED = RANDOM_SEED
ACTIVE_CONTEXT = ""

TUT_CODES = [
    "Greeting",
    "Instruction",
    "Guiding feedback",
    "Aligning to prior knowledge",
    "Understanding/Engagement-Tutor",
    "Encouragement",
]

TRANSCRIPT_FILES = [
    "First Author Copy GPT-Then-Human - Transcript B.csv",
    "First Author Copy GPT-Then-Human - transcript C.csv",
]
REQUIRED_DATA = TRANSCRIPT_FILES + ["datapoint_table.csv", "rationale_cache.csv"]

SIGNAL_MAP: dict[str, list[str]] = {
    "label": ["label"],
    "reasoning": ["reasoning"],
    "ambiguity": ["ambiguity"],
    "label_reasoning": ["label", "reasoning"],
    "label_ambiguity": ["label", "ambiguity"],
    "reasoning_ambiguity": ["reasoning", "ambiguity"],
    "all_three": ["label", "reasoning", "ambiguity"],
}

RESULT_FIELDS = [
    "round",
    "pooled_kappa",
    "overall_acc",
    "UE_kappa",
    "UE_TP",
    "UE_FP",
    "UE_FN",
    "UE_TN",
    "codebook_word_count",
    "n_changes",
    "n_signal_cases",
]

from consensus_coding import extract_and_complete_code, generate


def set_global_seed(seed: int) -> None:
    global ACTIVE_SEED
    ACTIVE_SEED = seed
    random.seed(seed)
    np.random.seed(seed)


set_global_seed(RANDOM_SEED)


# ── data validation ───────────────────────────────────────────────────────────

def check_data() -> None:
    missing = [f for f in REQUIRED_DATA if not (DATA / f).exists()]
    if missing:
        print("FATAL: missing required data files in ./data/:")
        for f in missing:
            print(f"  - {f}")
        print("\nSee data/README.txt for instructions.")
        sys.exit(1)


def list_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"FATAL: cannot reach Ollama ({e}). Start with: ollama serve")
        sys.exit(1)


def require_models() -> None:
    models = list_ollama_models()
    for name in (MODEL_A, MODEL_B, REVISER_MODEL):
        ok = any(m == name or m.startswith(name) for m in models)
        if not ok:
            print(f"FATAL: model '{name}' not available. Run: ollama pull {name}")
            print(f"Available: {models}")
            sys.exit(1)
        print(f"  OK: {name}")


def load_codebook(path: Path) -> tuple[dict[str, str], str, str]:
    """Return (codes dict, optional context string, source filename)."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    cb = obj["codes"]
    for k in TUT_CODES:
        if k not in cb:
            print(f"FATAL: codebook missing code '{k}' in {path.name}")
            sys.exit(1)
    codes = {k: (cb[k] if cb[k] is not None else "") for k in TUT_CODES}
    context = (obj.get("context") or "").strip()
    return codes, context, path.name


# ── tutoring rows + gold ────────────────────────────────────────────────────────

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


def load_tutor_rows() -> list[dict]:
    records: list[dict] = []
    for fname in TRANSCRIPT_FILES:
        path = DATA / fname
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
                text = (values[ti] or "").strip()
                if not text:
                    continue
                labels = {
                    c: human_label_for_code(header, values, c) for c in TUT_CODES
                }
                records.append({"text": text, "labels": labels})
    for i, r in enumerate(records, 1):
        r["item_id"] = f"tuto_{i:04d}"
    return records


# ── signal metadata (static pools) ────────────────────────────────────────────

def load_signal_pools() -> dict:
    dp = pd.read_csv(DATA / "datapoint_table.csv")
    dp_tut = dp[dp["dataset"] == "tutoring"]
    utt = (
        dp_tut.groupby("item_id")
        .agg(
            any_label_disagree=("label_disagree", "max"),
            n_codes_disagree=("label_disagree", "sum"),
            utt_ambiguity=("ambiguity", "mean"),
            reasoning_disagree=("reasoning_disagree", "first"),
        )
        .reset_index()
    )
    meta = utt.set_index("item_id").to_dict("index")

    amb_eligible = utt.dropna(subset=["utt_ambiguity"])
    n_matched = len(amb_eligible)
    if n_matched == 0:
        print("FATAL: no ambiguity-eligible tutoring utterances in datapoint_table.csv")
        sys.exit(1)

    reasoning_ids = (
        utt.sort_values("reasoning_disagree", ascending=False, na_position="last")
        .head(n_matched)["item_id"]
        .tolist()
    )
    ambiguity_ids = (
        amb_eligible.sort_values("utt_ambiguity", ascending=False)
        .head(n_matched)["item_id"]
        .tolist()
    )
    label_template_ids = (
        utt[utt["any_label_disagree"] == 1]
        .sort_values(
            ["n_codes_disagree", "utt_ambiguity", "item_id"],
            ascending=[False, False, True],
        )
        .head(n_matched)["item_id"]
        .tolist()
    )

    return {
        "n_matched": n_matched,
        "reasoning_ids": set(reasoning_ids),
        "ambiguity_ids": set(ambiguity_ids),
        "label_template_ids": set(label_template_ids),
        "meta": meta,
    }


def round_label_ids(
    dual: dict[str, dict], records: list[dict], n_matched: int
) -> list[str]:
    ranked: list[tuple] = []
    for r in records:
        iid = r["item_id"]
        a, b = dual[iid]["agentA"], dual[iid]["agentB"]
        if any(a.get(c, 0) != b.get(c, 0) for c in TUT_CODES):
            n_dis = sum(1 for c in TUT_CODES if a.get(c, 0) != b.get(c, 0))
            ranked.append((iid, n_dis))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return [iid for iid, _ in ranked[:n_matched]]


def select_signal_cases(
    condition: str,
    round_label: list[str],
    pools: dict,
) -> list[str]:
    signals = SIGNAL_MAP[condition]
    n_matched = pools["n_matched"]
    meta = pools["meta"]
    round_label_set = set(round_label)

    selected: set[str] = set()
    for sig in signals:
        if sig == "label":
            selected.update(round_label_set if round_label else pools["label_template_ids"])
        elif sig == "reasoning":
            selected.update(pools["reasoning_ids"])
        elif sig == "ambiguity":
            selected.update(pools["ambiguity_ids"])

    if len(selected) <= n_matched:
        return sorted(selected)

    def priority(iid: str) -> tuple:
        sig_hits = sum(
            1
            for sig in signals
            if (
                (sig == "label" and iid in round_label_set)
                or (sig == "reasoning" and iid in pools["reasoning_ids"])
                or (sig == "ambiguity" and iid in pools["ambiguity_ids"])
            )
        )
        m = meta.get(iid, {})
        rd = float(m.get("reasoning_disagree") or 0)
        amb = float(m.get("utt_ambiguity") or 0)
        return (sig_hits, rd, amb, iid)

    ranked = sorted(selected, key=priority, reverse=True)
    return ranked[:n_matched]


# ── coding + metrics ──────────────────────────────────────────────────────────

def codebook_text(cb: dict[str, str], context: str = "") -> str:
    lines: list[str] = []
    ctx = (context or ACTIVE_CONTEXT or "").strip()
    if ctx:
        lines.append(f"TASK CONTEXT: {ctx}\n")
    lines.extend(
        [
            "The codes below describe TUTOR moves in online tutoring dialogue.",
            "They are NOT mutually exclusive: an utterance can have several codes, or none.",
            "Mark 1 if the utterance fits the code, 0 otherwise.",
            "If a code has no definition yet, infer from the code NAME and task context only.\n",
        ]
    )
    for code, defn in cb.items():
        shown = defn.strip() if (defn or "").strip() else "(no definition yet)"
        lines.append(f"{code}: {shown}\n")
    return "\n".join(lines).strip()


def make_system_prompt(cb: dict[str, str], agent_name: str) -> str:
    codes_str = str(TUT_CODES)
    example_str = str({k: 0 for k in TUT_CODES})
    return (
        f"You are {agent_name}, a careful qualitative coding agent.\n"
        f"{codebook_text(cb)}\n\n"
        "Always write your reasoning FIRST (1–2 plain sentences). "
        "ONLY AFTER your reasoning, on the very last line, output a Python dictionary "
        f"with exactly these keys {codes_str}, values 0 or 1 only, no markdown, no "
        f"code fences. Example:\n{example_str}"
    )


def code_utterance(text: str, cb: dict[str, str], model: str, agent_name: str) -> dict:
    msgs = [
        {"role": "system", "content": make_system_prompt(cb, agent_name)},
        {"role": "user", "content": f"Utterance to code:\n{text}"},
    ]
    raw = generate(model, msgs, {"temperature": TEMP, "seed": ACTIVE_SEED})
    return extract_and_complete_code(raw, {k: 0 for k in TUT_CODES})


def word_count(cb: dict[str, str]) -> int:
    return sum(len(d.split()) for d in cb.values())


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    p1 = (tp + fp) / n
    p0 = (tn + fn) / n
    q1 = (tp + fn) / n
    q0 = (tn + fp) / n
    pe = p1 * q1 + p0 * q0
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def conf_counts(records, pred, code: str) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for r in records:
        p = pred[r["item_id"]].get(code, 0)
        g = r["labels"].get(code, 0)
        if p == 1 and g == 1:
            tp += 1
        elif p == 1 and g == 0:
            fp += 1
        elif p == 0 and g == 1:
            fn += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def kappa_report(records, pred) -> tuple[float, dict[str, float]]:
    tp = fp = fn = tn = 0
    per: dict[str, dict] = {c: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for c in TUT_CODES}
    for r in records:
        iid = r["item_id"]
        for code in TUT_CODES:
            p = pred[iid].get(code, 0)
            g = r["labels"].get(code, 0)
            bucket = per[code]
            if p == 1 and g == 1:
                tp += 1
                bucket["TP"] += 1
            elif p == 1 and g == 0:
                fp += 1
                bucket["FP"] += 1
            elif p == 0 and g == 1:
                fn += 1
                bucket["FN"] += 1
            else:
                tn += 1
                bucket["TN"] += 1
    pooled = cohen_kappa(tp, fp, fn, tn)
    per_k = {
        c: cohen_kappa(v["TP"], v["FP"], v["FN"], v["TN"]) for c, v in per.items()
    }
    return pooled, per_k


def accuracy_report(records, pred) -> float:
    ok = n = 0
    for r in records:
        iid = r["item_id"]
        for code in TUT_CODES:
            ok += int(pred[iid].get(code, 0) == r["labels"].get(code, 0))
            n += 1
    return ok / n if n else 0.0


# ── caches ────────────────────────────────────────────────────────────────────

def load_dual_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = row["item_id"]
            a = {c: int(row.get(f"A_{c}", 0) or 0) for c in TUT_CODES}
            b = {c: int(row.get(f"B_{c}", 0) or 0) for c in TUT_CODES}
            out[iid] = {"agentA": a, "agentB": b}
    return out


def save_dual_cache(path: Path, records, dual) -> None:
    fields = ["item_id", "text"] + [f"A_{c}" for c in TUT_CODES] + [f"B_{c}" for c in TUT_CODES]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            iid = r["item_id"]
            d = dual[iid]
            row = {"item_id": iid, "text": r["text"]}
            row.update({f"A_{c}": d["agentA"][c] for c in TUT_CODES})
            row.update({f"B_{c}": d["agentB"][c] for c in TUT_CODES})
            w.writerow(row)


def load_scored_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["item_id"]] = {c: int(row.get(c, 0) or 0) for c in TUT_CODES}
    return out


def save_scored_cache(path: Path, records, pred) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "text"] + TUT_CODES)
        w.writeheader()
        for r in records:
            iid = r["item_id"]
            row = {"item_id": iid, "text": r["text"]}
            row.update({c: pred[iid].get(c, 0) for c in TUT_CODES})
            w.writerow(row)


def run_dual_agents(records, cb, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"    Reusing cache: {cache_path.name}")
        return load_dual_cache(cache_path)
    dual: dict[str, dict] = {}
    n = len(records)
    print(f"    Running BOTH agents on {n} utterances (2×{n} LLM calls) ...")
    for i, r in enumerate(records, 1):
        if i % 25 == 0 or i == 1:
            print(f"      dual-agent {i}/{n}")
        a = code_utterance(r["text"], cb, MODEL_A, "Agent A")
        b = code_utterance(r["text"], cb, MODEL_B, "Agent B")
        dual[r["item_id"]] = {"agentA": a, "agentB": b}
    save_dual_cache(cache_path, records, dual)
    print(f"    Saved {cache_path.name}")
    return dual


def run_agent_a_scoring(records, cb, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"    Reusing scored cache: {cache_path.name}")
        return load_scored_cache(cache_path)
    pred: dict[str, dict] = {}
    n = len(records)
    print(f"    Scoring agent A on {n} utterances ...")
    for i, r in enumerate(records, 1):
        if i % 50 == 0 or i == 1:
            print(f"      agent A {i}/{n}")
        pred[r["item_id"]] = code_utterance(r["text"], cb, MODEL_A, "Agent A")
    save_scored_cache(cache_path, records, pred)
    print(f"    Saved {cache_path.name}")
    return pred


# ── reviser ───────────────────────────────────────────────────────────────────

def build_reviser_blocks(
    dual: dict, records: list[dict], case_ids: list[str]
) -> tuple[str, str]:
    case_set = set(case_ids)
    agreed_pos: dict[str, list[str]] = {c: [] for c in TUT_CODES}
    for r in records:
        iid = r["item_id"]
        a, b = dual[iid]["agentA"], dual[iid]["agentB"]
        for code in TUT_CODES:
            if a.get(code, 0) == 1 and b.get(code, 0) == 1:
                agreed_pos[code].append(r["text"])

    anchor_lines = [
        "Typical cases both coders agreed on (preserve these):",
        "(Utterances where BOTH Agent A and Agent B assigned the code = 1.",
        " No human labels were used.)",
    ]
    for code in TUT_CODES:
        pool = agreed_pos[code]
        if not pool:
            anchor_lines.append(f"\n{code}: (no agreed-positive cases found)")
            continue
        sample = random.sample(pool, min(ANCHOR_SAMPLE, len(pool)))
        anchor_lines.append(
            f"\n{code} ({len(pool)} agreed positives; showing up to {ANCHOR_SAMPLE}):"
        )
        for t in sample:
            anchor_lines.append(f'  - "{t}"')

    case_lines = ["Signal-selected cases for this revision round:"]
    shown = 0
    id_to_row = {r["item_id"]: r for r in records}
    for i, iid in enumerate(case_ids, 1):
        if iid not in id_to_row:
            continue
        r = id_to_row[iid]
        a, b = dual[iid]["agentA"], dual[iid]["agentB"]
        split = [c for c in TUT_CODES if a.get(c) != b.get(c)]
        case_lines.append(
            f'[{i}] "{r["text"]}"\n'
            f"    Agent A (Qwen) labels:  {a}\n"
            f"    Agent B (Llama) labels: {b}\n"
            f"    Codes where they split: {split if split else '(none this round)'}"
        )
        shown += 1
    if shown == 0:
        case_lines.append("(no cases matched — check signal selection)")

    return "\n".join(anchor_lines), "\n".join(case_lines)


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    m = re.search(r"\{[\s\S]*\}", text)
    candidate = m.group() if m else text
    candidate = candidate.replace("\\'", "'")
    candidate = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", candidate)
    return json.loads(candidate)


def call_reviser(
    current_cb: dict[str, str],
    typical_block: str,
    cases_block: str,
    round_num: int,
    condition: str,
) -> tuple[dict[str, str], list[str]]:
    cb_block = "\n".join(
        f"  {k}: {v if (v or '').strip() else '(no definition yet)'}"
        for k, v in current_cb.items()
    )
    ctx = (ACTIVE_CONTEXT or "").strip()
    context_block = f"TASK CONTEXT: {ctx}\n\n" if ctx else ""
    bare = all(not (v or "").strip() for v in current_cb.values())
    if bare:
        core_rule = (
            "- If a definition is blank, draft a SHORT initial definition from the code NAME, "
            "task context, and typical agreed examples — then keep edits minimal thereafter.\n"
            "- Do not invent theory beyond what the name and examples support."
        )
    else:
        core_rule = (
            "- Preserve each code's core meaning (e.g. Understanding/Engagement-Tutor = tutor "
            "checks understanding, usually by asking a question)."
        )
    prompt = f"""You are helping two coders agree on a tutoring dialogue codebook using
deductive coding. BUILD ON the existing codebook below. Make the SMALLEST edits needed to
reduce disagreement on the cases shown — clarify wording, add a short include/exclude note,
or add ONE example phrase. Do NOT rewrite or redefine a code; do not change its core
meaning. Keep all 6 code names. Prefer ADDING a sentence over rewriting one.

{context_block}Signal condition for this run: {condition}
You are given ONLY the two agents' labels — you do NOT have human gold labels.

CURRENT CODEBOOK:
{cb_block}

--- (A) Typical cases both coders agreed on (preserve these) ---
{typical_block}

--- (B) Signal-selected cases (reduce disagreement on these) ---
{cases_block}

CONSERVATIVE RULES:
- Every definition MUST still clearly cover its typical agreed examples above.
{core_rule}
- Make minimal, targeted edits only for the specific confusions in (B).
- At most one short include/exclude note and/or one example per code, drawn from the
  TYPICAL agreed set in (A), not from disagreement cases.
- Do not merge, delete, or rename codes.

Return ONLY valid JSON (no markdown, no code fences, no extra text):
{{
  "codes": {{
    "Greeting": "<definition — build on current, minimal edit>",
    "Instruction": "<definition — build on current, minimal edit>",
    "Guiding feedback": "<definition — build on current, minimal edit>",
    "Aligning to prior knowledge": "<definition — build on current, minimal edit>",
    "Understanding/Engagement-Tutor": "<definition — build on current, minimal edit>",
    "Encouragement": "<definition — build on current, minimal edit>"
  }},
  "changes": ["<short note per actual edit made>"]
}}"""

    print(f"    Calling reviser ({REVISER_MODEL}) for round {round_num} ...")
    msgs = [{"role": "user", "content": prompt}]
    raw = generate(REVISER_MODEL, msgs, {"temperature": TEMP, "seed": ACTIVE_SEED})
    print(f"    Reviser output (first 400 chars): {raw[:400].replace(chr(10), ' ')}")

    try:
        obj = extract_json(raw)
        for k in TUT_CODES:
            if k not in obj.get("codes", {}):
                raise KeyError(f"missing code '{k}'")
    except Exception as e:
        print(f"    Parse failed ({e}). Re-prompting once ...")
        retry = msgs + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Return ONLY valid JSON with 'codes' and 'changes' keys. Nothing else."
                ),
            },
        ]
        raw = generate(REVISER_MODEL, retry, {"temperature": TEMP, "seed": ACTIVE_SEED})
        obj = extract_json(raw)
        for k in TUT_CODES:
            if k not in obj.get("codes", {}):
                print(f"FATAL: could not parse revision for round {round_num}")
                print(raw)
                sys.exit(1)

    revised = {k: obj["codes"][k] for k in TUT_CODES}
    return revised, obj.get("changes", [])


# ── results I/O ───────────────────────────────────────────────────────────────

def append_result_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


def trend_read(rows: list[dict]) -> str:
    kappas = [float(r["pooled_kappa"]) for r in rows]
    if len(kappas) < 2:
        return "insufficient rounds"
    start, final = kappas[0], kappas[-1]
    delta = final - start
    if delta >= 0.02:
        return f"pooled kappa RISES ({start:.4f} -> {final:.4f}, delta={delta:+.4f})"
    if delta <= -0.02:
        return f"pooled kappa FALLS ({start:.4f} -> {final:.4f}, delta={delta:+.4f})"
    return f"pooled kappa mixed/plateau ({start:.4f} -> {final:.4f}, delta={delta:+.4f})"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Part 3 multi-round codebook revision")
    parser.add_argument(
        "--condition",
        required=True,
        choices=sorted(SIGNAL_MAP.keys()),
        help="Signal condition for reviser case selection",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="RNG seed (default 42)")
    parser.add_argument(
        "--outdir",
        default="",
        help="Output folder relative to package root (default: results/<condition>)",
    )
    parser.add_argument(
        "--codebook",
        default="codebook_gpt_human.json",
        help="Codebook JSON in package root (default: codebook_gpt_human.json)",
    )
    args = parser.parse_args()
    condition = args.condition
    set_global_seed(args.seed)

    global ACTIVE_CONTEXT
    cb_path = Path(args.codebook)
    if not cb_path.is_absolute():
        cb_path = PKG / cb_path
    if not cb_path.exists():
        print(f"FATAL: codebook not found: {cb_path}")
        sys.exit(1)
    theory_cb, ACTIVE_CONTEXT, cb_source = load_codebook(cb_path)

    if args.outdir:
        out_dir = PKG / args.outdir
    else:
        out_dir = RESULTS_ROOT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    summary_path = out_dir / "summary.txt"

    print("=" * 72)
    print(f"PART 3 EXPERIMENT — condition={condition} | TUTORING | kappa metric")
    print(f"Random seed: {args.seed} (Python random + numpy + Ollama options.seed)")
    print(f"Round-0 codebook: {cb_source}")
    if ACTIVE_CONTEXT:
        print(f"Task context: {ACTIVE_CONTEXT}")
    print(f"Output: {out_dir}")
    print("=" * 72)

    check_data()
    print("\nChecking Ollama models ...")
    require_models()

    tutor_rows = load_tutor_rows()
    pools = load_signal_pools()
    print(f"\nLoaded {len(tutor_rows)} tutor utterances")
    print(f"N_matched (signal case target): {pools['n_matched']}")
    print(f"Signals: {SIGNAL_MAP[condition]}")

    round0_path = out_dir / "codebook_round0.json"
    round0_payload = {
        "codes": theory_cb,
        "changes": [],
        "source": cb_source,
    }
    if ACTIVE_CONTEXT:
        round0_payload["context"] = ACTIVE_CONTEXT
    round0_path.write_text(json.dumps(round0_payload, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    current_cb = dict(theory_cb)

    # Round 0
    print("\n" + "=" * 72)
    print(f"ROUND 0 — CODEBOOK ({cb_source})")
    print("=" * 72)
    scored0 = out_dir / "scored_round0_baseline.csv"
    pred0 = run_agent_a_scoring(tutor_rows, theory_cb, scored0)
    pooled, per_k = kappa_report(tutor_rows, pred0)
    acc = accuracy_report(tutor_rows, pred0)
    ue = conf_counts(tutor_rows, pred0, FOCUS)
    row0 = {
        "round": 0,
        "pooled_kappa": f"{pooled:.4f}",
        "overall_acc": f"{acc:.4f}",
        "UE_kappa": f"{per_k[FOCUS]:.4f}",
        "UE_TP": ue["TP"],
        "UE_FP": ue["FP"],
        "UE_FN": ue["FN"],
        "UE_TN": ue["TN"],
        "codebook_word_count": word_count(theory_cb),
        "n_changes": 0,
        "n_signal_cases": 0,
    }
    append_result_row(results_csv, row0)
    all_rows.append(row0)
    print(
        f"    Round 0: pooled_kappa={pooled:.4f}  acc={acc:.4f}  "
        f"UE_kappa={per_k[FOCUS]:.4f}  words={word_count(theory_cb)}"
    )

    for r in range(1, N_ROUNDS + 1):
        print("\n" + "=" * 72)
        print(f"ROUND {r} — input codebook from round {r - 1}")
        print("=" * 72)

        dual_path = out_dir / f"dual_agent_round{r}_input.csv"
        dual = run_dual_agents(tutor_rows, current_cb, dual_path)

        r_label = round_label_ids(dual, tutor_rows, pools["n_matched"])
        case_ids = select_signal_cases(condition, r_label, pools)
        print(f"    Signal cases shown to reviser: {len(case_ids)}")

        typical, cases = build_reviser_blocks(dual, tutor_rows, case_ids)
        rev_path = out_dir / f"revised_codebook_round{r}.json"
        if rev_path.exists():
            print(f"    Reusing {rev_path.name}")
            parsed = json.loads(rev_path.read_text(encoding="utf-8"))
            revised_cb = {k: parsed["codes"][k] for k in TUT_CODES}
            changes = parsed.get("changes", [])
        else:
            revised_cb, changes = call_reviser(
                current_cb, typical, cases, r, condition
            )
            rev_path.write_text(
                json.dumps({"codes": revised_cb, "changes": changes}, indent=2),
                encoding="utf-8",
            )
            print(f"    Saved {rev_path.name} ({len(changes)} changes)")

        scored_path = out_dir / f"scored_round{r}.csv"
        pred_r = run_agent_a_scoring(tutor_rows, revised_cb, scored_path)
        pooled, per_k = kappa_report(tutor_rows, pred_r)
        acc = accuracy_report(tutor_rows, pred_r)
        ue = conf_counts(tutor_rows, pred_r, FOCUS)
        row_r = {
            "round": r,
            "pooled_kappa": f"{pooled:.4f}",
            "overall_acc": f"{acc:.4f}",
            "UE_kappa": f"{per_k[FOCUS]:.4f}",
            "UE_TP": ue["TP"],
            "UE_FP": ue["FP"],
            "UE_FN": ue["FN"],
            "UE_TN": ue["TN"],
            "codebook_word_count": word_count(revised_cb),
            "n_changes": len(changes),
            "n_signal_cases": len(case_ids),
        }
        append_result_row(results_csv, row_r)
        all_rows.append(row_r)
        print(
            f"    Round {r}: pooled_kappa={pooled:.4f}  acc={acc:.4f}  "
            f"UE_kappa={per_k[FOCUS]:.4f}  words={word_count(revised_cb)}  "
            f"changes={len(changes)}"
        )
        current_cb = revised_cb

    hdr = (
        f"{'round':>5} {'pool_k':>8} {'acc':>8} {'UE_k':>8} {'UE_TP':>6} "
        f"{'words':>6} {'cases':>6} {'chg':>4}"
    )
    lines = [
        f"Part 3 Experiment — condition={condition}",
        f"Random seed: {args.seed}",
        f"Reviser: {REVISER_MODEL} | Coders: {MODEL_A} + {MODEL_B}",
        f"Round-0 codebook: {cb_source} ({word_count(theory_cb)} definition words)",
        f"Task context: {ACTIVE_CONTEXT or '(none)'}",
        f"Signals: {SIGNAL_MAP[condition]} | N_matched={pools['n_matched']}",
        "",
        hdr,
        "-" * len(hdr),
    ]
    print("\n" + "=" * 72)
    print("ROUND-BY-ROUND TABLE")
    print("=" * 72)
    print(hdr)
    print("-" * len(hdr))
    for row in all_rows:
        line = (
            f"{int(row['round']):>5} {row['pooled_kappa']:>8} {row['overall_acc']:>8} "
            f"{row['UE_kappa']:>8} {row['UE_TP']:>6} "
            f"{row['codebook_word_count']:>6} {row['n_signal_cases']:>6} "
            f"{row['n_changes']:>4}"
        )
        print(line)
        lines.append(line)

    read = trend_read(all_rows)
    print(f"\nTREND: {read}")
    lines += ["", f"TREND: {read}"]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {summary_path.name}")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
