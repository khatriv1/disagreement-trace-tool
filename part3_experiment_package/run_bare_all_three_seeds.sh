#!/usr/bin/env bash
# Bare codebook all_three — three seeds sequential (42, 123, 7).
set -euo pipefail
cd "$(dirname "$0")"
LOG="bare_all_three_run.log"
PYTHON="${PYTHON:-../.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

SEEDS=(42 123 7)
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "BARE CODEBOOK all_three — seeds ${SEEDS[*]}"
echo "Started: $(date)"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
  OUT="results_bare_all_three_seed${SEED}"
  echo ""
  echo "############################################################"
  echo "# SEED ${SEED} — started $(date)"
  echo "# Outdir: ${OUT}"
  echo "############################################################"
  "$PYTHON" run_experiment.py \
    --condition all_three \
    --seed "$SEED" \
    --outdir "$OUT" \
    --codebook codebook_bare.json
  echo ""
  echo "# Significance for seed ${SEED} ..."
  "$PYTHON" significance.py --results-dir "$OUT"
  echo "# SEED ${SEED} — FINISHED $(date)"
done

echo ""
echo "============================================================"
echo "COMBINED TABLE — $(date)"
echo "============================================================"

"$PYTHON" << 'PYEOF'
import csv
import re
from pathlib import Path

PKG = Path(".")
seeds = [42, 123, 7]
print(f"{'seed':>6} {'r0':>8} {'best':>8} {'best_r':>7} {'r5':>8} {'delta':>9} {'CI_lo':>9} {'CI_hi':>9} {'p':>8}")
print("-" * 80)
rising = []
for s in seeds:
    d = PKG / f"results_bare_all_three_seed{s}"
    rows = list(csv.DictReader((d / "results.csv").open()))
    r0 = float(rows[0]["pooled_kappa"])
    best_r, best_k = 0, -1.0
    for row in rows:
        r = int(row["round"])
        if r == 0:
            continue
        k = float(row["pooled_kappa"])
        if k > best_k:
            best_k, best_r = k, r
    r5 = float(rows[-1]["pooled_kappa"])
    delta = r5 - r0
    sig = (d / "significance_results.txt").read_text()
    # first Round 0 vs Round 5 block
    m_ci = re.search(
        r"Round 0 vs Round 5.*?Bootstrap 95% CI: \[([+\-0-9.]+), ([+\-0-9.]+)\]",
        sig,
        re.S,
    )
    m_p = re.search(
        r"Round 0 vs Round 5.*?Permutation p \(two-sided\): ([0-9.]+)",
        sig,
        re.S,
    )
    ci_lo = float(m_ci.group(1)) if m_ci else float("nan")
    ci_hi = float(m_ci.group(2)) if m_ci else float("nan")
    p = float(m_p.group(1)) if m_p else float("nan")
    print(
        f"{s:>6} {r0:>8.4f} {best_k:>8.4f} {best_r:>7} {r5:>8.4f} "
        f"{delta:>+9.4f} {ci_lo:>+9.4f} {ci_hi:>+9.4f} {p:>8.4f}"
    )
    rising.append(delta >= 0.02 and ci_lo > 0)

n_rise = sum(1 for x in rising if x)
if n_rise == 3:
    verdict = "YES — bare start shows kappa RISING (reliable improvement) in ALL 3 seeds"
elif n_rise > 0:
    verdict = f"PARTIAL — reliable rise in {n_rise}/3 seeds (CI entirely above 0 and Δ≥0.02)"
elif any(float(list(csv.DictReader((PKG / f'results_bare_all_three_seed{s}' / 'results.csv').open()))[-1]['pooled_kappa']) >
         float(list(csv.DictReader((PKG / f'results_bare_all_three_seed{s}' / 'results.csv').open()))[0]['pooled_kappa'])
         for s in seeds):
    verdict = "POINT estimates rise in some seeds, but NOT reliably (CI spans 0 or small Δ)"
else:
    verdict = "NO — bare start does NOT show kappa rising over rounds across these seeds"

print()
print("VERDICT:", verdict)
summary = Path("results_bare_all_three_summary.txt")
# rebuild printable table for file
lines = [
    "Bare codebook all_three — 3 seeds",
    f"{'seed':>6} {'r0':>8} {'best':>8} {'best_r':>7} {'r5':>8} {'delta':>9} {'CI_lo':>9} {'CI_hi':>9} {'p':>8}",
    "-" * 80,
]
for s in seeds:
    d = PKG / f"results_bare_all_three_seed{s}"
    rows = list(csv.DictReader((d / "results.csv").open()))
    r0 = float(rows[0]["pooled_kappa"])
    best_r, best_k = 0, -1.0
    for row in rows:
        r = int(row["round"])
        if r == 0:
            continue
        k = float(row["pooled_kappa"])
        if k > best_k:
            best_k, best_r = k, r
    r5 = float(rows[-1]["pooled_kappa"])
    delta = r5 - r0
    sig = (d / "significance_results.txt").read_text()
    m_ci = re.search(
        r"Round 0 vs Round 5.*?Bootstrap 95% CI: \[([+\-0-9.]+), ([+\-0-9.]+)\]",
        sig,
        re.S,
    )
    m_p = re.search(
        r"Round 0 vs Round 5.*?Permutation p \(two-sided\): ([0-9.]+)",
        sig,
        re.S,
    )
    ci_lo = float(m_ci.group(1)) if m_ci else float("nan")
    ci_hi = float(m_ci.group(2)) if m_ci else float("nan")
    p = float(m_p.group(1)) if m_p else float("nan")
    lines.append(
        f"{s:>6} {r0:>8.4f} {best_k:>8.4f} {best_r:>7} {r5:>8.4f} "
        f"{delta:>+9.4f} {ci_lo:>+9.4f} {ci_hi:>+9.4f} {p:>8.4f}"
    )
lines += ["", f"VERDICT: {verdict}"]
summary.write_text("\n".join(lines) + "\n")
print(f"\nSaved {summary}")
PYEOF

rm -f results_bare_all_three.zip
zip -r results_bare_all_three.zip \
  results_bare_all_three_seed42 \
  results_bare_all_three_seed123 \
  results_bare_all_three_seed7 \
  results_bare_all_three_summary.txt \
  codebook_bare.json
echo ""
echo "Created results_bare_all_three.zip ($(du -h results_bare_all_three.zip | cut -f1))"
echo "ALL DONE — $(date)"
