#!/usr/bin/env python3
"""
Data-point level mixed-effects model comparing label_disagree, reasoning_disagree,
and per-item ambiguity as predictors of LLM correctness.

This is the proper unit of analysis (each utterance x code decision = 1 row),
following the approach in Conrad's temperature and persona paper, which models
each coding decision with random intercepts for item and code.

Outcome: llm_correct = did the LLM's two-agent decision match the human gold label?
  Rule: when both agents agree on a code, that is the LLM decision.
        when agents disagree, we set llm_correct = 1 if EITHER agent matches the
        human label (lenient/optimistic rule; documented below).
        Alternative (strict) rule explored in a robustness note at the end.

Inputs: rationale_cache.csv, lak24-coded-utterances.csv, tutoring transcript CSVs.
Does NOT re-run any LLM agents.

Note: the mixed-effects models use BinomialBayesMixedGLM from statsmodels.
"""

import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHEM_CSV = "lak24-coded-utterances.csv"
TUT_FILES = [
    "Data_2/First Author Copy GPT-Then-Human - Transcript B.csv",
    "Data_2/First Author Copy GPT-Then-Human - transcript C.csv",
]
CACHE_CSV = "rationale_cache.csv"
OUT_TABLE = "datapoint_table.csv"

CHEM_CODES = ["process", "plan", "act", "wrong"]
# 6 tutoring codes with kappa only
TUT_CODES = [
    "Greeting",
    "Instruction",
    "Guiding feedback",
    "Aligning to prior knowledge",
    "Understanding/Engagement-Tutor",
    "Encouragement",
]


# ---------------------------------------------------------------------------
# Gold label loading
# ---------------------------------------------------------------------------
def parse01_chem(value: str) -> int:
    """Chemistry gold labels are 'Yes'/'No'."""
    return 1 if (value or "").strip().lower() == "yes" else 0


def parse01_tut(value: str) -> int:
    s = str(value).strip().casefold()
    return 1 if s in ("1", "1.0", "true", "yes") else 0


def load_chem_gold(base: Path) -> list[dict]:
    rows = []
    with (base / CHEM_CSV).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("remove_flag", "").strip().lower() == "yes":
                continue
            text = (r.get("utterance_combined") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    **{c: parse01_chem(r.get(c, "No")) for c in CHEM_CODES},
                }
            )
    return rows


def human_label_for_code(header: list[str], values: list[str], code: str) -> int:
    target = code.strip().casefold()
    matched = []
    for name, val in zip(header, values):
        if (name or "").strip().casefold() == target:
            matched.append(parse01_tut(val))
    return max(matched) if matched else 0


def load_tut_gold(base: Path) -> list[dict]:
    rows = []
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
                    values = values + [""] * (len(header) - len(values))
                if (values[si] or "").strip().casefold() != "tutor":
                    continue
                text = (values[ti] or "").strip()
                if not text:
                    continue
                rows.append(
                    {
                        "text": text,
                        **{
                            c: human_label_for_code(header, values, c)
                            for c in TUT_CODES
                        },
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------
def load_cache(base: Path) -> list[dict]:
    rows = []
    with (base / CACHE_CSV).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = json.loads(r["agentA_labels"])
            b = json.loads(r["agentB_labels"])
            rows.append(
                {
                    "dataset": r["dataset"],
                    "idx": int(r["idx"]),
                    "text": r["text"],
                    "agentA_labels": a,
                    "agentB_labels": b,
                    "agentA_rationale": r["agentA_rationale"],
                    "agentB_rationale": r["agentB_rationale"],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Embeddings + per-data-point ambiguity
# ---------------------------------------------------------------------------
def cosine_safe(u: list[float], v: list[float]) -> float:
    a, b = np.array(u, dtype=float), np.array(v, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_per_item_reasoning_disagree(
    cache_rows: list[dict],
) -> dict[tuple[str, int], float]:
    print(f"\nEmbedding {len(cache_rows)} rationale pairs ...")
    from consensus_coding import embed

    texts = [r["agentA_rationale"] for r in cache_rows] + [
        r["agentB_rationale"] for r in cache_rows
    ]
    all_emb = embed(texts)
    n = len(cache_rows)
    emb_a = all_emb[:n]
    emb_b = all_emb[n:]

    out = {}
    for r, u, v in zip(cache_rows, emb_a, emb_b):
        sim = cosine_safe(u, v)
        if math.isnan(sim):
            sim = 0.0
        out[(r["dataset"], r["idx"])] = 1.0 - sim
    print(f"  Done. Mean reasoning_disagree = {np.mean(list(out.values())):.4f}")
    return out


def compute_per_datapoint_ambiguity(
    cache_rows: list[dict],
    codes_by_dataset: dict[str, list[str]],
    gold_by_dataset: dict[str, list[dict]],
) -> dict[tuple[str, int, str], float]:
    """
    For each (dataset, idx, code): max cosine similarity from this utterance to
    any utterance NOT labeled with that code (negative set), using the utterance
    text embeddings. Higher = harder boundary.
    """
    from consensus_coding import embed

    print("\nComputing per-data-point ambiguity ...")
    amb: dict[tuple[str, int, str], float] = {}

    for dataset, codes in codes_by_dataset.items():
        gold = gold_by_dataset[dataset]
        rows_ds = [r for r in cache_rows if r["dataset"] == dataset]
        n = len(rows_ds)
        print(f"  [{dataset}] embedding {n} utterance texts ...")
        texts = [r["text"] for r in rows_ds]
        embs = np.array(embed(texts), dtype=float)

        # normalise rows
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        embs_n = embs / norms

        for code in codes:
            labels = np.array(
                [gold[r["idx"] - 1].get(code, 0) for r in rows_ds], dtype=int
            )
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                for r in rows_ds:
                    amb[(dataset, r["idx"], code)] = float("nan")
                continue
            # cosine similarity matrix pos x neg
            sim_mat = embs_n[pos_idx] @ embs_n[neg_idx].T  # (n_pos, n_neg)
            max_sim = sim_mat.max(axis=1)  # (n_pos,)
            pos_to_max = dict(zip(pos_idx.tolist(), max_sim.tolist()))
            for i, r in enumerate(rows_ds):
                if i in pos_to_max:
                    amb[(dataset, r["idx"], code)] = float(pos_to_max[i])
                else:
                    # negative examples: ambiguity not defined the same way; use nan
                    amb[(dataset, r["idx"], code)] = float("nan")

    n_valid = sum(1 for v in amb.values() if not math.isnan(v))
    print(
        f"  Ambiguity computed for {n_valid}/{len(amb)} (dataset, item, code) combos."
    )
    return amb


# ---------------------------------------------------------------------------
# Build long table
# ---------------------------------------------------------------------------
def as01(v) -> int:
    return 1 if v == 1 or v == "1" else 0


def build_long_table(
    cache_rows: list[dict],
    gold_by_dataset: dict[str, list[dict]],
    codes_by_dataset: dict[str, list[str]],
    reasoning_disagree: dict[tuple[str, int], float],
    ambiguity: dict[tuple[str, int, str], float],
) -> list[dict]:
    """
    One row per (utterance, code).

    llm_correct rule (lenient/optimistic):
      - If both agents agree: llm_correct = 1 iff their shared label == human_label.
      - If agents disagree: llm_correct = 1 if EITHER agent matches human_label.
        Rationale: when there is agent disagreement we want to ask whether the system
        can still get the answer right via either reasoning path; the strict rule
        (require both) would conflate disagreement with error by construction.
    """
    rows = []
    for cache_row in cache_rows:
        ds = cache_row["dataset"]
        idx = cache_row["idx"]
        gold_list = gold_by_dataset[ds]
        gold = gold_list[idx - 1]
        codes = codes_by_dataset[ds]
        rd_val = reasoning_disagree.get((ds, idx), float("nan"))

        for code in codes:
            a = as01(cache_row["agentA_labels"].get(code, 0))
            b = as01(cache_row["agentB_labels"].get(code, 0))
            h = gold.get(code, 0)
            label_dis = int(a != b)

            # lenient rule
            if a == b:
                llm_correct = int(a == h)
            else:
                llm_correct = int(a == h or b == h)

            amb_val = ambiguity.get((ds, idx, code), float("nan"))

            rows.append(
                {
                    "dataset": ds,
                    "item_id": f"{ds[:4]}_{idx:04d}",
                    "code": code,
                    "human_label": h,
                    "agentA_label": a,
                    "agentB_label": b,
                    "label_disagree": label_dis,
                    "reasoning_disagree": rd_val,
                    "ambiguity": amb_val,
                    "llm_correct": llm_correct,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Mixed-effects models via BinomialBayesMixedGLM
# ---------------------------------------------------------------------------
def fit_model(df, formula: str, random_effects: dict) -> object:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    model = BinomialBayesMixedGLM.from_formula(
        formula, random_effects, df
    )
    result = model.fit_map()
    return result


def lr_test(lp1: float, lp2: float) -> tuple[float, float]:
    """LR = 2 * (log-posterior_2 - log-posterior_1), chi2(df=1).
    We use the MAP log-posterior as a proxy; differences between nested models
    cancel the prior contribution to first order."""
    lr = 2.0 * (lp2 - lp1)
    p = float(chi2.sf(lr, df=1))
    return lr, p


def _run_lme4(csv_path: str, r_script: str) -> str:
    """Write r_script to a temp file and run it with Rscript; return stdout."""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(r_script)
        rpath = f.name
    result = subprocess.run(
        ["Rscript", "--vanilla", rpath],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Rscript failed:\n{result.stderr}")
    return result.stdout


def run_models(df) -> None:
    """
    Mixed-effects logistic regression via lme4 glmer (called through Rscript subprocess).
    Two random intercepts: item_id (utterance) AND code.

    Including item_id is essential: reasoning_disagree is computed once per utterance
    and repeated across that utterance's codes. Without an item-level random intercept
    the repeated value is treated as independent across codes, inflating its apparent
    contribution. item_id absorbs all utterance-level variance so the fixed effects
    capture only within-utterance, cross-code variation.

    Fitting note: BinomialBayesMixedGLM (statsmodels) fails with singular Hessian at
    ~1003 item_id levels (too many parameters for MAP inversion). lme4 uses adaptive
    Gauss-Hermite quadrature / PIRLS and handles this cleanly. pymer4/rpy2 fails due
    to an R version mismatch (rpy2 built for R 4.5-arm64, R 4.4.2 installed), so we
    call Rscript via subprocess directly.

    Outcome: llm_correct_A = 1 if agentA (Qwen) matches human gold label.
    """
    import tempfile, os
    import pandas as pd
    from scipy.stats import spearmanr, pearsonr
    import statsmodels.formula.api as smf

    # ── Outcome ───────────────────────────────────────────────────────────────
    df = df.copy()
    df["llm_correct_A"] = (
        df["agentA_label"].astype(int) == df["human_label"].astype(int)
    ).astype(int)

    print("\n" + "=" * 78)
    print("OUTCOME: llm_correct_A  (agentA = Qwen vs human gold label)")
    print("=" * 78)
    vc_counts = df["llm_correct_A"].value_counts().sort_index()
    for v, c in vc_counts.items():
        print(f"  {v}: {c}  ({100*c/len(df):.1f}%)")

    # ── Predictor correlations ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PREDICTOR CORRELATIONS (label_disagree vs other predictors)")
    print("=" * 78)
    for pred in ["reasoning_disagree", "ambiguity"]:
        sub = df.dropna(subset=[pred])
        ld = sub["label_disagree"].astype(float).values
        px = sub[pred].astype(float).values
        sr, sp = spearmanr(ld, px)
        pr, pp = pearsonr(ld, px)
        print(
            f"  label_disagree vs {pred}: "
            f"Spearman r={sr:+.3f} p={sp:.4f}  "
            f"Pearson r={pr:+.3f} p={pp:.4f}  (n={len(sub)})"
        )

    # ── z-score helper ────────────────────────────────────────────────────────
    def zscore(col: pd.Series) -> pd.Series:
        mu, sd = col.mean(), col.std()
        return (col - mu) / sd if sd != 0 else col - mu

    def sign_note(coef: float) -> str:
        if math.isnan(coef):
            return "  [sign unknown]"
        if coef < 0:
            return "  Sign is NEGATIVE ✓ (expected: more disagreement/ambiguity → less accurate)."
        return (
            "  *** RED FLAG: sign is POSITIVE — more disagreement/ambiguity predicts\n"
            "  *** HIGHER accuracy. Unexpected; likely a confound (see interpretation)."
        )

    # ── Build data subsets ────────────────────────────────────────────────────
    df_f = df.dropna(subset=["reasoning_disagree"]).copy()
    df_f["label_disagree_z"]     = zscore(df_f["label_disagree"].astype(float))
    df_f["reasoning_disagree_z"] = zscore(df_f["reasoning_disagree"].astype(float))

    df_a = df.dropna(subset=["reasoning_disagree", "ambiguity"]).copy()
    df_a["label_disagree_z"] = zscore(df_a["label_disagree"].astype(float))
    df_a["ambiguity_z"]      = zscore(df_a["ambiguity"].astype(float))

    # ── Write CSV for R ───────────────────────────────────────────────────────
    tmp_full = tempfile.NamedTemporaryFile(suffix="_full.csv", delete=False)
    tmp_amb  = tempfile.NamedTemporaryFile(suffix="_amb.csv",  delete=False)
    df_f.to_csv(tmp_full.name, index=False)
    df_a.to_csv(tmp_amb.name,  index=False)
    tmp_full.close(); tmp_amb.close()

    # ── R script template ─────────────────────────────────────────────────────
    # lme4 glmer with two random intercepts (item_id + code).
    # We use anova(m0, m1) for the likelihood-ratio test (ML not REML).
    R_TEMPLATE = r"""
suppressPackageStartupMessages({{
  library(lme4)
}})
cat("--- {label} ---\n")
df <- read.csv("{csv}")
ctrl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))

m0 <- glmer({outcome} ~ label_disagree_z + (1|item_id) + (1|code),
            data=df, family=binomial, control=ctrl)
mA <- glmer({outcome} ~ label_disagree_z + {pred}_z + (1|item_id) + (1|code),
            data=df, family=binomial, control=ctrl)

cat("\nModel 0 fixed effects:\n")
print(summary(m0)$coefficients, digits=4)
cat("\nModel augmented fixed effects:\n")
print(summary(mA)$coefficients, digits=4)

lrt <- anova(m0, mA, test="Chisq")
cat("\nLRT (anova):\n")
print(lrt)

coefs <- coef(summary(mA))
pred_name <- "{pred}_z"
if (pred_name %in% rownames(coefs)) {{
  row <- coefs[pred_name, ]
  cat(sprintf("\nKEY: %s  coef=%+.4f  SE=%.4f  z=%.4f  p=%.4f\n",
              pred_name, row["Estimate"], row["Std. Error"], row["z value"], row["Pr(>|z|)"]))
}} else {{
  cat("KEY: predictor not found in model\n")
}}
cat("LRT_P:", lrt[["Pr(>Chisq)"]][2], "\n")
"""

    # ── PART A: data-point level (lme4, item_id + code) ───────────────────────
    print("\n" + "=" * 78)
    print("PART A — DATA-POINT LEVEL  (lme4 glmer, random intercepts: item_id + code)")
    print(f"  Rows — full: {len(df_f)}, ambiguity subset: {len(df_a)}")
    print("=" * 78)

    def run_lme4_model(label, csv, outcome, pred):
        script = R_TEMPLATE.format(
            label=label, csv=csv, outcome=outcome, pred=pred
        )
        print(f"\n  Running {label} via Rscript ...")
        out = _run_lme4(csv, script)
        print(out)
        # parse LRT p
        for line in out.splitlines():
            if line.startswith("LRT_P:"):
                try:
                    p_lrt = float(line.split(":")[1].strip())
                except Exception:
                    p_lrt = float("nan")
                return out, p_lrt
        return out, float("nan")

    def parse_key_line(out: str, pred_z: str):
        """Extract (coef, SE, z, p) from KEY line printed by R script."""
        for line in out.splitlines():
            if line.startswith("KEY:") and pred_z in line:
                parts = line.split()
                try:
                    coef = float(parts[2].split("=")[1])
                    se   = float(parts[3].split("=")[1])
                    z    = float(parts[4].split("=")[1])
                    p    = float(parts[5].split("=")[1])
                    return coef, se, z, p
                except Exception:
                    pass
        return float("nan"), float("nan"), float("nan"), float("nan")

    def print_verdict(pred_label: str, coef: float, p_lrt: float) -> None:
        print(sign_note(coef))
        sig = (not math.isnan(p_lrt)) and p_lrt < 0.05
        sign_ok = (not math.isnan(coef)) and coef < 0
        verdict = "YES" if sig else "NO"
        sv = "sign EXPECTED ✓" if sign_ok else "sign UNEXPECTED ✗"
        print(
            f"  VERDICT ({pred_label}): {verdict} — "
            f"{'adds' if sig else 'does NOT add'} significantly beyond label_disagree "
            f"(LRT p={p_lrt:.4f}); {sv}."
        )

    # Model R
    out_r, p_lrt_r = run_lme4_model(
        "Model R (reasoning)", tmp_full.name, "llm_correct_A", "reasoning_disagree"
    )
    c_r, se_r, z_r, p_r_coef = parse_key_line(out_r, "reasoning_disagree_z")
    print(
        f"  reasoning_disagree: coef={c_r:+.4f}, SE={se_r:.4f}, "
        f"z={z_r:.4f}, p_coef={p_r_coef:.4f}"
    )
    print_verdict("reasoning", c_r, p_lrt_r)

    # Model A
    out_a, p_lrt_a = run_lme4_model(
        "Model A (ambiguity)", tmp_amb.name, "llm_correct_A", "ambiguity"
    )
    c_a, se_a, z_a, p_a_coef = parse_key_line(out_a, "ambiguity_z")
    print(
        f"  ambiguity: coef={c_a:+.4f}, SE={se_a:.4f}, "
        f"z={z_a:.4f}, p_coef={p_a_coef:.4f}"
    )
    print_verdict("ambiguity", c_a, p_lrt_a)

    # ── PART B: utterance-level cross-check ───────────────────────────────────
    print("\n" + "=" * 78)
    print("PART B — UTTERANCE-LEVEL CROSS-CHECK  (OLS, one row per utterance)")
    print(
        "  reasoning_disagree is intrinsically per-utterance (same value repeated\n"
        "  across that utterance's codes). Aggregating removes the pseudo-replication\n"
        "  concern. Outcome: proportion of codes where agentA matched human (z-scored)."
    )
    print("=" * 78)

    utt = (
        df_f.groupby("item_id")
        .agg(
            prop_correct_A      =("llm_correct_A",      "mean"),
            label_disagree_mean =("label_disagree",      "mean"),
            reasoning_disagree  =("reasoning_disagree",  "first"),
            dataset             =("dataset",             "first"),
        )
        .reset_index()
    )
    utt["prop_z"]          = zscore(utt["prop_correct_A"].astype(float))
    utt["label_z"]         = zscore(utt["label_disagree_mean"].astype(float))
    utt["reasoning_z"]     = zscore(utt["reasoning_disagree"].astype(float))
    print(f"\n  Utterance-level rows: {len(utt)}")

    m_u0 = smf.ols("prop_z ~ label_z",               data=utt).fit()
    m_ur = smf.ols("prop_z ~ label_z + reasoning_z",  data=utt).fit()

    lr_u, p_u = lr_test(m_u0.llf, m_ur.llf)
    c_u  = float(m_ur.params.get("reasoning_z", float("nan")))
    se_u = float(m_ur.bse.get("reasoning_z", float("nan")))
    p_u_coef = float(m_ur.pvalues.get("reasoning_z", float("nan")))

    print(f"\n  OLS Model 0  R²={m_u0.rsquared:.4f}")
    print(m_u0.summary().tables[1])
    print(f"\n  OLS Model R  R²={m_ur.rsquared:.4f}")
    print(m_ur.summary().tables[1])
    print(
        f"\n  reasoning_disagree (utterance-level): "
        f"coef={c_u:+.4f}, SE={se_u:.4f}, p={p_u_coef:.4f}"
    )
    print(sign_note(c_u))
    print(f"  LR test: LR={lr_u:.4f}, df=1, p={p_u:.4f}")
    sig_u = p_u < 0.05
    sign_u = c_u < 0
    print(
        f"  CROSS-CHECK VERDICT: reasoning_disagree "
        f"{'IS' if sig_u else 'IS NOT'} significant at utterance level "
        f"(LR p={p_u:.4f}); sign {'EXPECTED ✓' if sign_u else 'UNEXPECTED ✗'}."
    )

    # ── cleanup temp files ────────────────────────────────────────────────────
    os.unlink(tmp_full.name)
    os.unlink(tmp_amb.name)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(
        f"  Modelling: lme4 glmer (R via Rscript), family=binomial.\n"
        f"  Random intercepts: item_id (utterance) + code.\n"
        f"  n = {len(df_f)} data-point rows / {len(utt)} utterances.\n"
        f"  At this n even tiny effects reach significance; interpret effect sizes.\n"
        f"  llm_correct_A base rate: 87.1% — high ceiling limits sensitivity.\n"
        f"\n"
        f"  reasoning_disagree: coef={c_r:+.4f}, LRT p={p_lrt_r:.4f}  "
        f"[utterance-level cross-check LRT p={p_u:.4f}]\n"
        f"  ambiguity:          coef={c_a:+.4f}, LRT p={p_lrt_a:.4f}"
    )


# ---------------------------------------------------------------------------
# Main — loads from existing datapoint_table.csv; does NOT re-run agents
# ---------------------------------------------------------------------------
def main() -> None:
    import pandas as pd

    base = Path(__file__).resolve().parent
    table_path = base / OUT_TABLE

    if not table_path.exists():
        raise FileNotFoundError(
            f"{OUT_TABLE} not found. Run the script once without --fast to build it."
        )

    print(f"Loading {OUT_TABLE} ...")
    df = pd.read_csv(table_path)
    print(f"  Shape: {df.shape}  columns: {list(df.columns)}")

    print("\nSummary statistics:")
    for col in ("label_disagree", "reasoning_disagree", "ambiguity", "agentA_label",
                "agentB_label", "human_label", "llm_correct"):
        sub = df[col].dropna()
        print(
            f"  {col}: n={len(sub)} mean={sub.mean():.4f} sd={sub.std():.4f}"
        )

    run_models(df)


if __name__ == "__main__":
    main()
