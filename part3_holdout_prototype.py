#!/usr/bin/env python3
"""
part3_holdout_prototype.py
ONE-round Part 3 codebook revision with:
  (1) Bigger reviser: qwen2.5:14b
  (2) HOLDOUT design: reviser never sees human labels — only utterances + both
      agents' labels. Human gold is used ONLY for held-out scoring.

Condition: LABEL | Dataset: TUTORING | Round: 1
Coders: agent A = qwen2.5:7b (scoring); agent B labels come from datapoint_table.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent

MODEL_A = "qwen2.5:7b"
MODEL_B = "llama3.1:8b"  # labels already in datapoint_table; not re-run here
REVISER_MODEL = "qwen2.5:14b"
TEMP = 0.4

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

ORIGINAL_CODEBOOK: dict[str, str] = {
    "Greeting": "A salutation or farewell between tutor and student.",
    "Instruction": (
        "A specific instruction or direction the tutor gives about what to do."
    ),
    "Guiding feedback": (
        "Feedback on the student's work, or clarification/explanation of a concept; "
        "guiding the student through a problem."
    ),
    "Aligning to prior knowledge": (
        "The tutor points the student to a previously learned concept, often using "
        "'remember'."
    ),
    "Understanding/Engagement-Tutor": (
        "The tutor checks the student's understanding, usually by asking a question."
    ),
    "Encouragement": (
        "Affirmative statements praising the student's effort or performance."
    ),
}

from consensus_coding import extract_and_complete_code, generate


# ── Ollama model check ────────────────────────────────────────────────────────

def list_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=5
        ) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"FATAL: cannot reach Ollama ({e}). Start it with: ollama serve")
        sys.exit(1)


def require_model(name: str) -> None:
    models = list_ollama_models()
    # exact or tag-prefix match (e.g. qwen2.5:14b vs qwen2.5:14b-instruct)
    ok = any(m == name or m.startswith(name) for m in models)
    if not ok:
        print(f"FATAL: reviser model '{name}' is not available in Ollama.")
        print(f"Available models: {models}")
        sys.exit(1)
    print(f"  Reviser model '{name}' is available.")


# ── helpers ───────────────────────────────────────────────────────────────────

def codebook_text(cb: dict[str, str]) -> str:
    lines = [
        "The codes below describe TUTOR moves in online tutoring dialogue.",
        "They are NOT mutually exclusive: an utterance can have several codes, or none.",
        "Mark 1 if the utterance fits the code, 0 otherwise.\n",
    ]
    for code, defn in cb.items():
        lines.append(f"{code}: {defn}\n")
    return "\n".join(lines).strip()


def make_system_prompt(cb: dict[str, str]) -> str:
    codes_str = str(TUT_CODES)
    example_str = str({k: 0 for k in TUT_CODES})
    return (
        "You are Agent A, a careful qualitative coding agent.\n"
        f"{codebook_text(cb)}\n\n"
        "Always write your reasoning FIRST (1–2 plain sentences). "
        "ONLY AFTER your reasoning, on the very last line, output a Python dictionary "
        f"with exactly these keys {codes_str}, values 0 or 1 only, no markdown, no "
        f"code fences. Example:\n{example_str}"
    )


def code_utterance(text: str, cb: dict[str, str]) -> dict[str, int]:
    msgs = [
        {"role": "system", "content": make_system_prompt(cb)},
        {"role": "user", "content": f"Utterance to code:\n{text}"},
    ]
    raw = generate(MODEL_A, msgs, {"temperature": TEMP})
    return extract_and_complete_code(raw, {k: 0 for k in TUT_CODES})


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


def load_tutor_rows(base: Path) -> list[dict]:
    records: list[dict] = []
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
                text = (values[ti] or "").strip()
                if not text:
                    continue
                labels = {
                    c: human_label_for_code(header, values, c) for c in TUT_CODES
                }
                records.append({"text": text, "labels": labels})
    return records


def accuracy_report(
    records: list[dict], pred_key: str
) -> tuple[float, dict[str, float]]:
    total_correct = 0
    total_n = 0
    per_code: dict[str, float] = {}
    for code in TUT_CODES:
        correct = sum(
            1
            for r in records
            if r[pred_key].get(code, 0) == r["labels"].get(code, 0)
        )
        per_code[code] = correct / len(records) if records else 0.0
        total_correct += correct
        total_n += len(records)
    overall = total_correct / total_n if total_n else 0.0
    return overall, per_code


def conf_counts(records: list[dict], pred_key: str, code: str) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for r in records:
        p = r.get(pred_key, {}).get(code, 0)
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


def word_count(cb: dict[str, str]) -> int:
    return sum(len(d.split()) for d in cb.values())


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    m = re.search(r"\{[\s\S]*\}", text)
    candidate = m.group() if m else text
    candidate = candidate.replace("\\'", "'")
    candidate = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", candidate)
    return json.loads(candidate)


def try_parse(text: str) -> dict:
    obj = extract_json(text)
    if "codes" not in obj or not isinstance(obj["codes"], dict):
        raise ValueError("missing 'codes' key")
    for k in TUT_CODES:
        if k not in obj["codes"]:
            raise KeyError(f"missing code '{k}'")
    return obj


def agent_label_dicts(dp_rows) -> tuple[dict, dict]:
    """Build {code: 0/1} for agent A and B from datapoint rows for one item."""
    a, b = {}, {}
    for _, row in dp_rows.iterrows():
        code = row["code"]
        if code in TUT_CODES:
            a[code] = int(row["agentA_label"])
            b[code] = int(row["agentB_label"])
    return a, b


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("PART 3 HOLDOUT+ANCHORED PROTOTYPE — LABEL | TUTORING | reviser=qwen2.5:14b")
    print("Holdout: reviser sees model-agreed anchors + disagreements; NEVER human gold.")
    print("=" * 70)

    print("\nChecking Ollama models ...")
    require_model(REVISER_MODEL)
    require_model(MODEL_A)

    # ── STEP 0 — Baseline ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 0 — BASELINE (agent A, original codebook) [held-out scoring]")
    print("=" * 70)

    baseline_csv = BASE / "baseline_tutoring.csv"
    tutor_rows = load_tutor_rows(BASE)
    print(f"  Loaded {len(tutor_rows)} tutor utterances")

    if baseline_csv.exists():
        print(f"  Reusing existing {baseline_csv.name} ...")
        with baseline_csv.open(newline="", encoding="utf-8") as f:
            rows_bl = list(csv.DictReader(f))
        if len(rows_bl) != len(tutor_rows):
            print(
                f"  WARNING: baseline rows ({len(rows_bl)}) != gold ({len(tutor_rows)}); "
                "rebuilding baseline."
            )
            baseline_csv.unlink()
        else:
            for r_bl, r_gold in zip(rows_bl, tutor_rows):
                r_gold["pred_orig"] = {
                    c: int(r_bl[c]) for c in TUT_CODES if c in r_bl
                }

    if not baseline_csv.exists() or "pred_orig" not in tutor_rows[0]:
        print(f"  Running agent A on {len(tutor_rows)} utterances (original) ...")
        for i, r in enumerate(tutor_rows, 1):
            if i % 50 == 0 or i == 1:
                print(f"  ... {i}/{len(tutor_rows)}")
            r["pred_orig"] = code_utterance(r["text"], ORIGINAL_CODEBOOK)
        with baseline_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text"] + TUT_CODES)
            writer.writeheader()
            for r in tutor_rows:
                row = {"text": r["text"]}
                row.update({c: r["pred_orig"].get(c, 0) for c in TUT_CODES})
                writer.writerow(row)
        print(f"  Saved {baseline_csv.name}")

    baseline_overall, baseline_per_code = accuracy_report(tutor_rows, "pred_orig")
    print(
        f"\n  Baseline overall accuracy: {baseline_overall:.4f} "
        f"({baseline_overall*100:.1f}%)"
    )
    print("  Per-code:")
    for code in TUT_CODES:
        print(f"    {code:40s}: {baseline_per_code[code]:.4f}")

    # ── STEP 1 — Propose revision (holdout: agent labels only + agreed anchors) ─
    print("\n" + "=" * 70)
    print(
        f"STEP 1 — PROPOSE REVISED CODEBOOK  (LABEL holdout+anchors, reviser={REVISER_MODEL})"
    )
    print(
        "  Reviser input: (A) model-AGREED positives + (B) disagreement cases. "
        "NO human labels."
    )
    print("=" * 70)

    import random as _random

    import pandas as pd

    _random.seed(42)

    label_csv = BASE / "part3_label_tutoring.csv"
    rev_json = BASE / "revised_codebook_holdout_anchored_label_tutoring_round1.json"
    dp = pd.read_csv(BASE / "datapoint_table.csv")
    dp_tut = dp[dp["dataset"] == "tutoring"]

    # Map item_id -> text from cache / label cases / datapoint via part3 files
    text_by_id: dict[str, str] = {}
    with label_csv.open(newline="", encoding="utf-8") as f:
        label_cases = list(csv.DictReader(f))
    for c in label_cases:
        text_by_id[c["item_id"]] = c["text"]
    # Fill remaining tutoring texts from rationale_cache if needed
    cache_path = BASE / "rationale_cache.csv"
    if cache_path.exists():
        with cache_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["dataset"] != "tutoring":
                    continue
                iid = f"tuto_{int(r['idx']):04d}"
                if iid not in text_by_id:
                    text_by_id[iid] = r["text"]

    print(f"  Loaded {len(label_cases)} label-disagreement cases")

    # (A) Model-AGREED positive anchors: both agents assign code = 1
    agreed_pos: dict[str, list[str]] = {c: [] for c in TUT_CODES}
    for code in TUT_CODES:
        sub = dp_tut[
            (dp_tut["code"] == code)
            & (dp_tut["agentA_label"] == 1)
            & (dp_tut["agentB_label"] == 1)
        ]
        for iid in sub["item_id"].unique():
            t = text_by_id.get(iid)
            if t:
                agreed_pos[code].append(t)

    print("  Model-agreed positive counts (both agents = 1):")
    anchor_lines = [
        "Typical cases both coders agreed on (preserve these):",
        "(These are utterances where BOTH Agent A and Agent B assigned the code = 1.",
        " No human labels were used.)",
    ]
    for code in TUT_CODES:
        pool = agreed_pos[code]
        print(f"    {code}: {len(pool)}")
        if not pool:
            anchor_lines.append(f"\n{code}: (no agreed-positive cases found)")
            continue
        _random.shuffle(pool)
        sample = pool[:5]
        anchor_lines.append(
            f"\n{code} ({len(pool)} agreed positives; showing up to 5):"
        )
        for t in sample:
            anchor_lines.append(f'  - "{t}"')
    typical_block = "\n".join(anchor_lines)

    # (B) Disagreement cases
    cases_block_lines = [
        "Cases where the coders disagreed (reduce this):",
    ]
    missing = 0
    for i, c in enumerate(label_cases, 1):
        iid = c["item_id"]
        rows = dp_tut[dp_tut["item_id"] == iid]
        if rows.empty:
            missing += 1
            continue
        a_labs, b_labs = agent_label_dicts(rows)
        disagreed = [k for k in TUT_CODES if a_labs.get(k) != b_labs.get(k)]
        cases_block_lines.append(
            f"[{i}] \"{c['text']}\"\n"
            f"    Agent A (Qwen) labels:  {a_labs}\n"
            f"    Agent B (Llama) labels: {b_labs}\n"
            f"    Codes where they split: {disagreed}"
        )
    if missing:
        print(f"  WARNING: {missing} cases missing from datapoint_table")
    cases_block = "\n".join(cases_block_lines)

    cb_block = "\n".join(f"  {k}: {v}" for k, v in ORIGINAL_CODEBOOK.items())

    reviser_prompt = f"""Refine the codebook so the two coders agree more on the DISAGREEMENT
cases, WITHOUT losing the TYPICAL cases both coders already agreed on.

You are given ONLY the two agents' labels — you do NOT have human gold labels.

CURRENT CODEBOOK:
{cb_block}

--- (A) Typical cases both coders agreed on (preserve these) ---
{typical_block}

--- (B) Cases where the coders disagreed (reduce this) ---
{cases_block}

RULES:
- Every revised definition MUST still clearly cover its 'typical agreed' examples. Do
  not narrow or redirect a code so it would exclude its own typical examples.
- Preserve each code's core meaning (e.g. if a code is about the tutor asking a question
  to check understanding, keep that as the core).
- Add include/exclude boundaries only to resolve the specific confusions in the
  disagreement cases.
- Add 1-2 short example phrases per code drawn from the TYPICAL agreed set (not from the
  disagreement set).
- Keep the same 6 code names; do not merge or delete.

Return ONLY valid JSON (no markdown, no code fences, no extra text):
{{
  "codes": {{
    "Greeting": "<revised definition with include/exclude + examples>",
    "Instruction": "<revised definition with include/exclude + examples>",
    "Guiding feedback": "<revised definition with include/exclude + examples>",
    "Aligning to prior knowledge": "<revised definition with include/exclude + examples>",
    "Understanding/Engagement-Tutor": "<revised definition with include/exclude + examples>",
    "Encouragement": "<revised definition with include/exclude + examples>"
  }},
  "changes": ["<short note per code>"]
}}"""

    print(f"  Calling reviser ({REVISER_MODEL}) ... this may take a few minutes")
    msgs = [{"role": "user", "content": reviser_prompt}]
    raw_revision = generate(REVISER_MODEL, msgs, {"temperature": TEMP})
    print("  Raw reviser output (first 600 chars):")
    print("    " + raw_revision[:600].replace("\n", "\n    "))

    parsed = None
    try:
        parsed = try_parse(raw_revision)
        print("  Parsed successfully on first attempt.")
    except Exception as e:
        print(f"  Parse failed ({e}). Re-prompting once ...")
        retry_msgs = msgs + [
            {"role": "assistant", "content": raw_revision},
            {
                "role": "user",
                "content": (
                    "Your response was not valid JSON or was missing required keys. "
                    "Return ONLY the JSON object described above, nothing else."
                ),
            },
        ]
        raw2 = generate(REVISER_MODEL, retry_msgs, {"temperature": TEMP})
        print("  Re-prompt raw output (first 600 chars):")
        print("    " + raw2[:600].replace("\n", "\n    "))
        try:
            parsed = try_parse(raw2)
            print("  Parsed successfully on retry.")
        except Exception as e2:
            print(f"\nFATAL: Could not parse revised codebook after retry ({e2}).")
            print("Full second raw output:")
            print(raw2)
            sys.exit(1)

    revised_cb = {k: parsed["codes"][k] for k in TUT_CODES}
    changes = parsed.get("changes", [])

    rev_json.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"\n  Saved: {rev_json.name}")
    print(f"\n  Changes reported by reviser ({len(changes)}):")
    for i, ch in enumerate(changes, 1):
        print(f"    {i}. {ch}")
    print(f"\n  Original codebook word count: {word_count(ORIGINAL_CODEBOOK)}")
    print(f"  Revised codebook word count:  {word_count(revised_cb)}")

    # ── STEP 2 — Re-code ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — RE-CODE with revised codebook (agent A)")
    print("=" * 70)

    recoded_csv = BASE / "recoded_holdout_anchored_label_tutoring_round1.csv"
    print(f"  Running agent A on {len(tutor_rows)} utterances (revised) ...")
    for i, r in enumerate(tutor_rows, 1):
        if i % 50 == 0 or i == 1:
            print(f"  ... {i}/{len(tutor_rows)}")
        r["pred_rev"] = code_utterance(r["text"], revised_cb)

    with recoded_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text"] + TUT_CODES)
        writer.writeheader()
        for r in tutor_rows:
            row = {"text": r["text"]}
            row.update({c: r["pred_rev"].get(c, 0) for c in TUT_CODES})
            writer.writerow(row)
    print(f"  Saved: {recoded_csv.name}")

    revised_overall, revised_per_code = accuracy_report(tutor_rows, "pred_rev")

    # ── STEP 3 — Report ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3 — HELD-OUT RESULTS: ORIGINAL vs REVISED")
    print("=" * 70)

    col_w = 42
    print(f"  {'Code':<{col_w}} {'Original':>10} {'Revised':>10} {'Delta':>8}")
    print("  " + "-" * (col_w + 32))
    summary_lines = [
        "Part 3 Holdout Anchored Prototype — Round 1 Summary",
        f"Condition: LABEL | Dataset: TUTORING | Reviser: {REVISER_MODEL}",
        "Holdout: reviser saw model-AGREED anchors + disagreement cases; NO human gold.",
        f"N disagreement cases shown to reviser: {len(label_cases)}",
        f"N tutor utterances scored: {len(tutor_rows)}",
        "",
        f"{'Code':<42} {'Original':>10} {'Revised':>10} {'Delta':>8}",
        "-" * 72,
    ]
    for code in TUT_CODES:
        orig = baseline_per_code[code]
        rev = revised_per_code[code]
        delta = rev - orig
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"  {code:<{col_w}} {orig:>10.4f} {rev:>10.4f} {delta:>+7.4f} {arrow}")
        summary_lines.append(f"{code:<42} {orig:>10.4f} {rev:>10.4f} {delta:>+8.4f}")
    print("  " + "-" * (col_w + 32))
    overall_delta = revised_overall - baseline_overall
    arrow_ov = "▲" if overall_delta > 0 else ("▼" if overall_delta < 0 else "=")
    print(
        f"  {'OVERALL':<{col_w}} {baseline_overall:>10.4f} {revised_overall:>10.4f} "
        f"{overall_delta:>+7.4f} {arrow_ov}"
    )
    summary_lines += [
        "-" * 72,
        f"{'OVERALL':<42} {baseline_overall:>10.4f} {revised_overall:>10.4f} "
        f"{overall_delta:>+8.4f}",
        "",
        f"Codebook word count: {word_count(ORIGINAL_CODEBOOK)} -> {word_count(revised_cb)}",
        f"Changes by reviser: {len(changes)}",
        "",
        "Changes:",
    ]
    for i, ch in enumerate(changes, 1):
        summary_lines.append(f"  {i}. {ch}")

    print(
        f"\n  Codebook word count: {word_count(ORIGINAL_CODEBOOK)} → "
        f"{word_count(revised_cb)} "
        f"(delta {word_count(revised_cb)-word_count(ORIGINAL_CODEBOOK):+d})"
    )
    print(f"  Changes made by reviser: {len(changes)}")

    FOCUS = "Understanding/Engagement-Tutor"
    orig_c = conf_counts(tutor_rows, "pred_orig", FOCUS)
    rev_c = conf_counts(tutor_rows, "pred_rev", FOCUS)
    print(f"\n  Confusion matrix — '{FOCUS}':")
    print(f"  {'':6} {'Original':>10} {'Revised':>10} {'Delta':>8}")
    print("  " + "-" * 40)
    summary_lines += [
        "",
        f"Confusion matrix — {FOCUS}:",
        f"  {'':6} {'Original':>10} {'Revised':>10} {'Delta':>8}",
        "  " + "-" * 40,
    ]
    for key in ("TP", "FP", "FN", "TN"):
        d = rev_c[key] - orig_c[key]
        print(f"  {key:<6} {orig_c[key]:>10} {rev_c[key]:>10} {d:>+8}")
        summary_lines.append(f"  {key:<6} {orig_c[key]:>10} {rev_c[key]:>10} {d:>+8}")
    tp_preserved = rev_c["TP"]
    tp_total = orig_c["TP"]
    print(
        f"\n  True-positive preservation: {tp_preserved}/{tp_total} "
        f"({100*tp_preserved/max(tp_total,1):.1f}% of original TPs retained)"
    )
    summary_lines.append(
        f"  TP preserved: {tp_preserved}/{tp_total} "
        f"({100*tp_preserved/max(tp_total,1):.1f}%)"
    )

    summary_lines += ["", "Revised codebook definitions:"]
    for code, defn in revised_cb.items():
        summary_lines.append(f"\n{code}:\n  {defn}")

    summary_path = BASE / "part3_holdout_anchored_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n  Saved: {summary_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
