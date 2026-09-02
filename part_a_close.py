#!/usr/bin/env python3
"""
part_a_close.py
Three extra ambiguity-vs-label analyses to finish Part A (Conrad).

Reuses datapoint_table.csv. Does NOT re-run agents.
Outcome: llm_correct = agent A (Qwen) matches human gold.
Ambiguity is only defined for positively-labeled rows (n ≈ 582).
All model tests run on that SAME subset for a fair comparison.

1. Correlation: ambiguity vs label_disagree (raw + within-dataset z-score)
2. Does label_disagree add beyond ambiguity? (LRT nested models)
3. How predictive is ambiguity alone? (vs intercept-only)

Mixed models: lme4 glmer via Rscript, random intercepts for item_id + code.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent
TABLE = BASE / "datapoint_table.csv"
OUT_TXT = BASE / "part_a_close_results.txt"


def zscore_within(df: pd.DataFrame, col: str, group: str = "dataset") -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std()
        if sd == 0 or math.isnan(sd):
            return s - mu
        return (s - mu) / sd

    return df.groupby(group)[col].transform(_z)


def zscore(col: pd.Series) -> pd.Series:
    mu, sd = col.mean(), col.std()
    return (col - mu) / sd if sd != 0 else col - mu


def run_rscript(script: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(script)
        rpath = f.name
    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", rpath],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Rscript failed:\n{result.stderr}\n{result.stdout}")
        return result.stdout
    finally:
        os.unlink(rpath)


def parse_key(out: str, name: str) -> tuple[float, float, float, float]:
    """Parse KEY: name  coef=... SE=... z=... p=..."""
    for line in out.splitlines():
        if line.startswith("KEY:") and name in line:
            parts = line.split()
            try:
                coef = float(parts[2].split("=")[1])
                se = float(parts[3].split("=")[1])
                z = float(parts[4].split("=")[1])
                p = float(parts[5].split("=")[1])
                return coef, se, z, p
            except Exception:
                pass
    return float("nan"), float("nan"), float("nan"), float("nan")


def parse_lrt_p(out: str) -> float:
    for line in out.splitlines():
        if line.startswith("LRT_P:"):
            try:
                return float(line.split(":")[1].strip())
            except Exception:
                return float("nan")
    return float("nan")


def sign_note(coef: float) -> str:
    if math.isnan(coef):
        return "[sign unknown]"
    if coef < 0:
        return "NEGATIVE ✓ (expected: more disagreement/ambiguity → less correct)"
    return "POSITIVE ✗ (unexpected: more disagreement/ambiguity → more correct)"


def main() -> None:
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("PART A CLOSE — ambiguity vs label_disagree (decision-level)")
    emit("=" * 72)

    df = pd.read_csv(TABLE)
    # Outcome: agent A (Qwen) matches human (not the old lenient llm_correct column)
    df["llm_correct"] = (
        df["agentA_label"].astype(int) == df["human_label"].astype(int)
    ).astype(int)

    amb = df.dropna(subset=["ambiguity"]).copy()
    n = len(amb)
    emit(f"\nLoaded {TABLE.name}: {len(df)} rows total.")
    emit(
        f"Ambiguity-defined subset (positively-labeled decisions): n = {n}."
    )
    emit(
        "NOTE: all model tests below use this SAME ambiguity-defined subset "
        "for a fair comparison."
    )
    emit(
        f"llm_correct (agent A = human): mean = {amb['llm_correct'].mean():.4f}"
    )
    emit(
        f"label_disagree mean = {amb['label_disagree'].mean():.4f}; "
        f"ambiguity mean = {amb['ambiguity'].mean():.4f}"
    )

    # ── 1. Correlations ───────────────────────────────────────────────────────
    emit("\n" + "=" * 72)
    emit("1. CORRELATION: ambiguity vs label_disagree")
    emit("=" * 72)

    x = amb["ambiguity"].astype(float).values
    y = amb["label_disagree"].astype(float).values
    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
    emit(f"  Raw-pooled (n={n}):")
    emit(f"    Spearman r = {sr:+.4f}, p = {sp:.4f}")
    emit(f"    Pearson  r = {pr:+.4f}, p = {pp:.4f}")

    amb = amb.copy()
    amb["ambiguity_zds"] = zscore_within(amb, "ambiguity")
    amb["label_disagree_zds"] = zscore_within(amb, "label_disagree")
    xz = amb["ambiguity_zds"].values
    yz = amb["label_disagree_zds"].values
    sr_z, sp_z = spearmanr(xz, yz)
    pr_z, pp_z = pearsonr(xz, yz)
    emit("  Within-dataset z-scored, then pooled:")
    emit(f"    Spearman r = {sr_z:+.4f}, p = {sp_z:.4f}")
    emit(f"    Pearson  r = {pr_z:+.4f}, p = {pp_z:.4f}")

    # ── prepare model frame ───────────────────────────────────────────────────
    amb["ambiguity_z"] = zscore(amb["ambiguity"].astype(float))
    amb["label_disagree_z"] = zscore(amb["label_disagree"].astype(float))

    tmp = tempfile.NamedTemporaryFile(suffix="_amb.csv", delete=False)
    amb.to_csv(tmp.name, index=False)
    tmp.close()

    # ── 2. Label beyond ambiguity ─────────────────────────────────────────────
    emit("\n" + "=" * 72)
    emit("2. Does LABEL disagreement add BEYOND ambiguity?")
    emit("   Model 0: llm_correct ~ ambiguity + (1|item_id) + (1|code)")
    emit("   Model L: llm_correct ~ ambiguity + label_disagree + (1|item_id) + (1|code)")
    emit("=" * 72)

    r_script_2 = f"""
suppressPackageStartupMessages({{ library(lme4) }})
df <- read.csv("{tmp.name}")
ctrl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))

m0 <- glmer(llm_correct ~ ambiguity_z + (1|item_id) + (1|code),
            data=df, family=binomial, control=ctrl)
mL <- glmer(llm_correct ~ ambiguity_z + label_disagree_z + (1|item_id) + (1|code),
            data=df, family=binomial, control=ctrl)

cat("\\nModel 0 fixed effects:\\n")
print(summary(m0)$coefficients, digits=4)
cat("\\nModel L fixed effects:\\n")
print(summary(mL)$coefficients, digits=4)

lrt <- anova(m0, mL, test="Chisq")
cat("\\nLRT (anova):\\n")
print(lrt)

coefs <- coef(summary(mL))
row <- coefs["label_disagree_z", ]
cat(sprintf("\\nKEY: label_disagree_z  coef=%+.4f  SE=%.4f  z=%.4f  p=%.4f\\n",
            row["Estimate"], row["Std. Error"], row["z value"], row["Pr(>|z|)"]))
cat("LRT_P:", lrt[["Pr(>Chisq)"]][2], "\\n")
cat(sprintf("LL_0: %.6f\\n", as.numeric(logLik(m0))))
cat(sprintf("LL_L: %.6f\\n", as.numeric(logLik(mL))))
"""
    emit("\n  Fitting via lme4 (Rscript) ...")
    out2 = run_rscript(r_script_2)
    emit(out2)

    c_l, se_l, z_l, p_l = parse_key(out2, "label_disagree_z")
    p_lrt2 = parse_lrt_p(out2)
    emit(
        f"  label_disagree: coef={c_l:+.4f}, SE={se_l:.4f}, z={z_l:.4f}, p={p_l:.4f}"
    )
    emit(f"  Sign: {sign_note(c_l)}")
    emit(f"  LRT (Model L vs Model 0): p = {p_lrt2:.4f}")
    if (not math.isnan(p_lrt2)) and p_lrt2 < 0.05:
        verdict2 = (
            f"YES — label_disagree adds significantly beyond ambiguity "
            f"(LRT p={p_lrt2:.4f}); {sign_note(c_l)}"
        )
    else:
        verdict2 = (
            f"NO — label_disagree does NOT add significantly beyond ambiguity "
            f"(LRT p={p_lrt2:.4f})"
        )
    emit(f"  VERDICT: {verdict2}")

    # ── 3. Ambiguity alone ────────────────────────────────────────────────────
    emit("\n" + "=" * 72)
    emit("3. How predictive is AMBIGUITY ON ITS OWN?")
    emit("   Model null: llm_correct ~ 1 + (1|item_id) + (1|code)")
    emit("   Model A:    llm_correct ~ ambiguity + (1|item_id) + (1|code)")
    emit("=" * 72)

    r_script_3 = f"""
suppressPackageStartupMessages({{ library(lme4) }})
df <- read.csv("{tmp.name}")
ctrl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))

mNull <- glmer(llm_correct ~ 1 + (1|item_id) + (1|code),
               data=df, family=binomial, control=ctrl)
mA <- glmer(llm_correct ~ ambiguity_z + (1|item_id) + (1|code),
            data=df, family=binomial, control=ctrl)

cat("\\nModel null fixed effects:\\n")
print(summary(mNull)$coefficients, digits=4)
cat("\\nModel A fixed effects:\\n")
print(summary(mA)$coefficients, digits=4)

lrt <- anova(mNull, mA, test="Chisq")
cat("\\nLRT (anova):\\n")
print(lrt)

coefs <- coef(summary(mA))
row <- coefs["ambiguity_z", ]
cat(sprintf("\\nKEY: ambiguity_z  coef=%+.4f  SE=%.4f  z=%.4f  p=%.4f\\n",
            row["Estimate"], row["Std. Error"], row["z value"], row["Pr(>|z|)"]))
cat("LRT_P:", lrt[["Pr(>Chisq)"]][2], "\\n")

# McFadden-like pseudo-R2 using logLik vs null
ll_null <- as.numeric(logLik(mNull))
ll_a    <- as.numeric(logLik(mA))
pseudo_r2 <- 1 - (ll_a / ll_null)
cat(sprintf("LL_null: %.6f\\n", ll_null))
cat(sprintf("LL_A: %.6f\\n", ll_a))
cat(sprintf("PSEUDO_R2: %.6f\\n", pseudo_r2))
"""
    emit("\n  Fitting via lme4 (Rscript) ...")
    out3 = run_rscript(r_script_3)
    emit(out3)

    c_a, se_a, z_a, p_a = parse_key(out3, "ambiguity_z")
    p_lrt3 = parse_lrt_p(out3)
    pseudo_r2 = float("nan")
    for line in out3.splitlines():
        if line.startswith("PSEUDO_R2:"):
            try:
                pseudo_r2 = float(line.split(":")[1].strip())
            except Exception:
                pass

    emit(
        f"  ambiguity: coef={c_a:+.4f}, SE={se_a:.4f}, z={z_a:.4f}, p={p_a:.4f}"
    )
    emit(f"  Sign: {sign_note(c_a)}")
    emit(f"  LRT (Model A vs intercept-only): p = {p_lrt3:.4f}")
    emit(f"  McFadden-like pseudo-R² (vs null mixed model): {pseudo_r2:.4f}")
    if (not math.isnan(p_lrt3)) and p_lrt3 < 0.05:
        verdict3 = (
            f"YES — ambiguity alone significantly predicts correctness "
            f"(LRT p={p_lrt3:.4f}, pseudo-R²={pseudo_r2:.4f}); {sign_note(c_a)}"
        )
    else:
        verdict3 = (
            f"NO — ambiguity alone is NOT a significant predictor "
            f"(LRT p={p_lrt3:.4f}, pseudo-R²={pseudo_r2:.4f})"
        )
    emit(f"  VERDICT: {verdict3}")

    os.unlink(tmp.name)

    # ── Summary ───────────────────────────────────────────────────────────────
    emit("\n" + "=" * 72)
    emit("SUMMARY (plain language)")
    emit("=" * 72)
    emit(
        f"  1. Relatedness: ambiguity and label_disagree correlate "
        f"Spearman r={sr_z:+.3f} (z-scored within dataset, p={sp_z:.4f}). "
        + (
            "They overlap only weakly."
            if abs(sr_z) < 0.3
            else "They overlap moderately."
            if abs(sr_z) < 0.5
            else "They overlap substantially."
        )
    )
    emit(f"  2. Label beyond ambiguity: {verdict2}")
    emit(f"  3. Ambiguity alone: {verdict3}")
    emit(
        f"\n  n = {n} ambiguity-defined decisions; descriptive. "
        "Tiny effects can reach significance at this n — interpret effect sizes."
    )
    emit(
        "  Modelling: lme4 glmer (binomial), random intercepts for item_id + code."
    )

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    emit(f"\nSaved to {OUT_TXT.name}")


if __name__ == "__main__":
    main()
