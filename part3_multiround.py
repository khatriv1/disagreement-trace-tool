#!/usr/bin/env python3
"""
part3_multiround.py
Multi-round Part 3 codebook revision (LABEL | TUTORING), building on the working
anchored holdout prototype.

Design (unchanged from part3_holdout_prototype.py):
  - Coders: agent A = qwen2.5:7b, agent B = llama3.1:8b, temp 0.4
  - Reviser: qwen2.5:14b, temp 0.4
  - Holdout: reviser never sees human labels; only model-agreed anchors +
    label-disagreement cases from the CURRENT round's dual-agent labels

Round 0 = baseline (original codebook, agent A vs human gold).
Rounds 1..N_ROUNDS: dual-agent labels on current codebook -> reviser -> score revised.

All outputs go to part3_multiround_results/; zipped at the end.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "part3_multiround_results"

MODEL_A = "qwen2.5:7b"
MODEL_B = "llama3.1:8b"
REVISER_MODEL = "qwen2.5:14b"
TEMP = 0.4
N_ROUNDS = 5
FOCUS = "Understanding/Engagement-Tutor"
ANCHOR_SAMPLE = 5
RANDOM_SEED = 42

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

random.seed(RANDOM_SEED)

RESULT_FIELDS = [
    "round",
    "overall_acc",
    "UE_acc",
    "UE_TP",
    "UE_FP",
    "UE_FN",
    "UE_TN",
    "codebook_word_count",
    "n_changes",
]


# ── utilities ─────────────────────────────────────────────────────────────────

def list_ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=5
        ) as resp:
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
            print(f"FATAL: model '{name}' not available.")
            print(f"Available: {models}")
            sys.exit(1)
        print(f"  OK: {name}")


def codebook_text(cb: dict[str, str]) -> str:
    lines = [
        "The codes below describe TUTOR moves in online tutoring dialogue.",
        "They are NOT mutually exclusive: an utterance can have several codes, or none.",
        "Mark 1 if the utterance fits the code, 0 otherwise.\n",
    ]
    for code, defn in cb.items():
        lines.append(f"{code}: {defn}\n")
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
    raw = generate(model, msgs, {"temperature": TEMP})
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
    for i, r in enumerate(records, 1):
        r["item_id"] = f"tuto_{i:04d}"
    return records


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


def try_parse_revision(text: str) -> dict:
    obj = extract_json(text)
    if "codes" not in obj or not isinstance(obj["codes"], dict):
        raise ValueError("missing 'codes' key")
    for k in TUT_CODES:
        if k not in obj["codes"]:
            raise KeyError(f"missing code '{k}'")
    return obj


def accuracy_report(records: list[dict], pred: dict[str, dict]) -> tuple[float, dict]:
    total_correct = 0
    total_n = 0
    per_code: dict[str, float] = {}
    for code in TUT_CODES:
        correct = sum(
            1 for r in records if pred[r["item_id"]].get(code, 0) == r["labels"].get(code, 0)
        )
        per_code[code] = correct / len(records) if records else 0.0
        total_correct += correct
        total_n += len(records)
    return total_correct / total_n if total_n else 0.0, per_code


def conf_counts(
    records: list[dict], pred: dict[str, dict], code: str
) -> dict[str, int]:
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


def load_dual_cache(path: Path) -> dict[str, dict]:
    """Return item_id -> {agentA: {code:0/1}, agentB: {code:0/1}}."""
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = row["item_id"]
            a = {c: int(row.get(f"A_{c}", 0) or 0) for c in TUT_CODES}
            b = {c: int(row.get(f"B_{c}", 0) or 0) for c in TUT_CODES}
            out[iid] = {"agentA": a, "agentB": b, "text": row["text"]}
    return out


def save_dual_cache(path: Path, records: list[dict], dual: dict[str, dict]) -> None:
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


def save_scored_cache(path: Path, records: list[dict], pred: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "text"] + TUT_CODES)
        w.writeheader()
        for r in records:
            iid = r["item_id"]
            row = {"item_id": iid, "text": r["text"]}
            row.update({c: pred[iid].get(c, 0) for c in TUT_CODES})
            w.writerow(row)


def run_dual_agents(
    records: list[dict], cb: dict[str, str], cache_path: Path
) -> dict[str, dict]:
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
        dual[r["item_id"]] = {"agentA": a, "agentB": b, "text": r["text"]}
    save_dual_cache(cache_path, records, dual)
    print(f"    Saved {cache_path.name}")
    return dual


def run_agent_a_scoring(
    records: list[dict], cb: dict[str, str], cache_path: Path
) -> dict[str, dict]:
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


def build_reviser_blocks(dual: dict[str, dict], records: list[dict]) -> tuple[str, str, int]:
    """Return (typical_block, disagreement_block, n_disagree_utterances)."""
    # Agreed positives per code
    agreed_pos: dict[str, list[str]] = {c: [] for c in TUT_CODES}
    disagree_utts: list[dict] = []

    for r in records:
        iid = r["item_id"]
        d = dual[iid]
        a, b = d["agentA"], d["agentB"]
        text = r["text"]
        for code in TUT_CODES:
            if a.get(code, 0) == 1 and b.get(code, 0) == 1:
                agreed_pos[code].append(text)
        if any(a.get(c, 0) != b.get(c, 0) for c in TUT_CODES):
            disagree_utts.append(
                {"text": text, "agentA": a, "agentB": b, "item_id": iid}
            )

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

    disagree_lines = ["Cases where the coders disagreed (reduce this):"]
    for i, u in enumerate(disagree_utts, 1):
        split = [c for c in TUT_CODES if u["agentA"].get(c) != u["agentB"].get(c)]
        disagree_lines.append(
            f'[{i}] "{u["text"]}"\n'
            f"    Agent A (Qwen) labels:  {u['agentA']}\n"
            f"    Agent B (Llama) labels: {u['agentB']}\n"
            f"    Codes where they split: {split}"
        )

    return "\n".join(anchor_lines), "\n".join(disagree_lines), len(disagree_utts)


def call_reviser(
    current_cb: dict[str, str], typical_block: str, cases_block: str, round_num: int
) -> tuple[dict[str, str], list[str]]:
    cb_block = "\n".join(f"  {k}: {v}" for k, v in current_cb.items())
    prompt = f"""Refine the codebook so the two coders agree more on the DISAGREEMENT
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

    print(f"    Calling reviser ({REVISER_MODEL}) for round {round_num} ...")
    msgs = [{"role": "user", "content": prompt}]
    raw = generate(REVISER_MODEL, msgs, {"temperature": TEMP})
    print(f"    Reviser output (first 400 chars): {raw[:400].replace(chr(10), ' ')}")

    try:
        parsed = try_parse_revision(raw)
    except Exception as e:
        print(f"    Parse failed ({e}). Re-prompting once ...")
        retry = msgs + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Your response was not valid JSON or was missing required keys. "
                    "Return ONLY the JSON object described above, nothing else."
                ),
            },
        ]
        raw2 = generate(REVISER_MODEL, retry, {"temperature": TEMP})
        try:
            parsed = try_parse_revision(raw2)
        except Exception as e2:
            print(f"FATAL: could not parse revision for round {round_num} ({e2})")
            print(raw2)
            sys.exit(1)

    revised = {k: parsed["codes"][k] for k in TUT_CODES}
    changes = parsed.get("changes", [])
    return revised, changes


def append_result_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def record_metrics(
    round_num: int,
    records: list[dict],
    pred: dict[str, dict],
    cb: dict[str, str],
    n_changes: int,
    results_csv: Path,
) -> dict:
    overall, per_code = accuracy_report(records, pred)
    ue = conf_counts(records, pred, FOCUS)
    row = {
        "round": round_num,
        "overall_acc": f"{overall:.4f}",
        "UE_acc": f"{per_code[FOCUS]:.4f}",
        "UE_TP": ue["TP"],
        "UE_FP": ue["FP"],
        "UE_FN": ue["FN"],
        "UE_TN": ue["TN"],
        "codebook_word_count": word_count(cb),
        "n_changes": n_changes,
    }
    append_result_row(results_csv, row)
    print(
        f"    Round {round_num}: overall={overall:.4f}  UE={per_code[FOCUS]:.4f}  "
        f"UE TP/FP/FN/TN={ue['TP']}/{ue['FP']}/{ue['FN']}/{ue['TN']}  "
        f"words={word_count(cb)}  changes={n_changes}"
    )
    return row


def trend_read(rows: list[dict]) -> str:
    accs = [float(r["overall_acc"]) for r in rows]
    if len(accs) < 2:
        return "insufficient rounds to assess trend"
    peak = max(accs)
    peak_i = accs.index(peak)
    final = accs[-1]
    if peak_i < len(accs) - 1 and final < peak - 0.005:
        return (
            f"rises then falls — peak overall_acc={peak:.4f} at round {rows[peak_i]['round']}, "
            f"final={final:.4f} (possible overfitting)"
        )
    if all(accs[i] <= accs[i + 1] + 0.001 for i in range(len(accs) - 1)):
        return f"keeps rising or flat — final overall_acc={final:.4f}"
    if abs(final - accs[-2]) < 0.003 and abs(accs[-2] - accs[-3]) < 0.003:
        return f"plateaus near {final:.4f} in last rounds"
    return f"mixed — peak={peak:.4f}, final={final:.4f}"


def zip_results(out_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out_dir.rglob("*")):
            if fp.is_file() and fp.name != ".DS_Store":
                zf.write(fp, fp.relative_to(out_dir.parent))
    print(f"\nCreated zip: {zip_path} ({zip_path.stat().st_size // 1024} KB)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results_csv = OUT / "part3_multiround_results.csv"
    summary_path = OUT / "part3_multiround_summary.txt"

    print("=" * 70)
    print("PART 3 MULTI-ROUND — LABEL | TUTORING | holdout+anchors")
    print(f"Rounds: 0 (baseline) + {N_ROUNDS} revision rounds")
    print(f"Output folder: {OUT}")
    print("=" * 70)

    print("\nChecking Ollama models ...")
    require_models()

    tutor_rows = load_tutor_rows(BASE)
    print(f"\nLoaded {len(tutor_rows)} tutor utterances")

    all_rows: list[dict] = []
    summary_lines: list[str] = [
        "Part 3 Multi-Round — LABEL | TUTORING",
        f"Reviser: {REVISER_MODEL} | Coders: {MODEL_A} + {MODEL_B}",
        f"Rounds: 0 baseline + {N_ROUNDS} revisions",
        "",
    ]

    current_cb = dict(ORIGINAL_CODEBOOK)

    # ── Round 0: baseline ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROUND 0 — BASELINE (original codebook, agent A)")
    print("=" * 70)

    scored0_path = OUT / "scored_round0_baseline.csv"
    baseline_outer = BASE / "baseline_tutoring.csv"

    if scored0_path.exists():
        pred0 = load_scored_cache(scored0_path)
    elif baseline_outer.exists():
        print("  Loading baseline from baseline_tutoring.csv ...")
        pred0 = {}
        with baseline_outer.open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                iid = f"tuto_{i:04d}"
                pred0[iid] = {c: int(row.get(c, 0) or 0) for c in TUT_CODES}
        save_scored_cache(scored0_path, tutor_rows, pred0)
    else:
        pred0 = run_agent_a_scoring(tutor_rows, ORIGINAL_CODEBOOK, scored0_path)

    row0 = record_metrics(0, tutor_rows, pred0, ORIGINAL_CODEBOOK, 0, results_csv)
    all_rows.append(row0)

    # Save round-0 codebook snapshot
    (OUT / "revised_codebook_round0.json").write_text(
        json.dumps({"codes": ORIGINAL_CODEBOOK, "changes": []}, indent=2),
        encoding="utf-8",
    )

    # ── Rounds 1..N ─────────────────────────────────────────────────────────
    for r in range(1, N_ROUNDS + 1):
        print("\n" + "=" * 70)
        print(f"ROUND {r} — input codebook from round {r-1}")
        print("=" * 70)
        print(f"  Input codebook word count: {word_count(current_cb)}")

        dual_path = OUT / f"dual_agent_round{r}_input.csv"
        dual = run_dual_agents(tutor_rows, current_cb, dual_path)

        typical_block, cases_block, n_dis = build_reviser_blocks(dual, tutor_rows)
        print(f"    Disagreement utterances this round: {n_dis}")

        rev_path = OUT / f"revised_codebook_round{r}.json"
        if rev_path.exists():
            print(f"    Reusing existing {rev_path.name}")
            parsed = json.loads(rev_path.read_text(encoding="utf-8"))
            revised_cb = {k: parsed["codes"][k] for k in TUT_CODES}
            changes = parsed.get("changes", [])
        else:
            revised_cb, changes = call_reviser(
                current_cb, typical_block, cases_block, r
            )
            rev_path.write_text(
                json.dumps({"codes": revised_cb, "changes": changes}, indent=2),
                encoding="utf-8",
            )
            print(f"    Saved {rev_path.name} ({len(changes)} changes)")

        scored_path = OUT / f"scored_round{r}.csv"
        pred_r = run_agent_a_scoring(tutor_rows, revised_cb, scored_path)

        row_r = record_metrics(
            r, tutor_rows, pred_r, revised_cb, len(changes), results_csv
        )
        all_rows.append(row_r)

        current_cb = revised_cb

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROUND-BY-ROUND TABLE")
    print("=" * 70)
    hdr = (
        f"{'round':>5} {'overall':>8} {'UE_acc':>8} {'UE_TP':>6} {'UE_FP':>6} "
        f"{'UE_FN':>6} {'UE_TN':>6} {'words':>6} {'chg':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    summary_lines.append(hdr)
    summary_lines.append("-" * len(hdr))
    for row in all_rows:
        line = (
            f"{int(row['round']):>5} {row['overall_acc']:>8} {row['UE_acc']:>8} "
            f"{row['UE_TP']:>6} {row['UE_FP']:>6} {row['UE_FN']:>6} "
            f"{row['UE_TN']:>6} {row['codebook_word_count']:>6} {row['n_changes']:>4}"
        )
        print(line)
        summary_lines.append(line)

    read = trend_read(all_rows)
    print(f"\nTREND: {read}")
    summary_lines += ["", f"TREND: {read}"]

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nSaved {summary_path.name}")

    zip_path = BASE / "part3_multiround_results.zip"
    zip_results(OUT, zip_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
