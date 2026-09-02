#!/usr/bin/env bash
# Resume/finish bare all_three: seed 7 only, then combined table + zip.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-../.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python3"
LOG="bare_seed7_finish.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Finishing seed 7 — $(date) ==="

# Clean CSV so re-run does not duplicate round-0 rows (caches preserved)
OUT="results_bare_all_three_seed7"
if [[ -f "$OUT/results.csv" ]]; then
  head -2 "$OUT/results.csv" > "$OUT/results.csv.tmp"
  mv "$OUT/results.csv.tmp" "$OUT/results.csv"
fi

"$PYTHON" run_experiment.py \
  --condition all_three \
  --seed 7 \
  --outdir "$OUT" \
  --codebook codebook_bare.json

# Dedupe results.csv (restart may append duplicate round-0 rows)
"$PYTHON" << 'DEDUPE'
import csv
from pathlib import Path
p = Path("results_bare_all_three_seed7/results.csv")
rows = list(csv.DictReader(p.open()))
by_round = {}
for row in rows:
    by_round[int(row["round"])] = row
with p.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    for r in sorted(by_round):
        w.writerow(by_round[r])
print(f"Deduped results.csv -> {len(by_round)} rounds")
DEDUPE

echo "=== Significance seed 7 — $(date) ==="
"$PYTHON" significance.py --results-dir "$OUT"

echo ""
echo "=== COMBINED TABLE (seeds 42, 123, 7) — $(date) ==="
"$PYTHON" << 'PYEOF'
import csv
import re
from pathlib import Path

seeds = [42, 123, 7]
print(f"{'seed':>6} {'r0':>8} {'best':>8} {'best_r':>7} {'r5':>8} {'delta':>9} {'CI_lo':>9} {'CI_hi':>9} {'p':>8}")
print("-" * 80)
reliable_rise = []
for s in seeds:
    d = Path(f"results_bare_all_three_seed{s}")
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
        sig, re.S,
    )
    m_p = re.search(
        r"Round 0 vs Round 5.*?Permutation p \(two-sided\): ([0-9.]+)",
        sig, re.S,
    )
    ci_lo = float(m_ci.group(1)) if m_ci else float("nan")
    ci_hi = float(m_ci.group(2)) if m_ci else float("nan")
    p = float(m_p.group(1)) if m_p else float("nan")
    print(
        f"{s:>6} {r0:>8.4f} {best_k:>8.4f} {best_r:>7} {r5:>8.4f} "
        f"{delta:>+9.4f} {ci_lo:>+9.4f} {ci_hi:>+9.4f} {p:>8.4f}"
    )
    reliable_rise.append(delta >= 0.02 and ci_lo > 0)

n = sum(reliable_rise)
if n == 3:
    verdict = "YES — kappa reliably RISES (r0→r5) in ALL 3 seeds"
elif n > 0:
    verdict = f"PARTIAL — reliable r0→r5 rise in {n}/3 seeds only"
elif any(
    float(list(csv.DictReader(open(f"results_bare_all_three_seed{s}/results.csv")))[-1]["pooled_kappa"])
    > float(list(csv.DictReader(open(f"results_bare_all_three_seed{s}/results.csv")))[0]["pooled_kappa"])
    for s in seeds
):
    verdict = "Some seeds rise at r5 but NOT reliably (CI spans 0)"
else:
    verdict = "NO — bare revision does NOT reliably raise kappa at r5"
print()
print("VERDICT:", verdict)

lines = ["Bare codebook all_three — 3 seeds", "-" * 80]
for s in seeds:
    d = Path(f"results_bare_all_three_seed{s}")
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
    sig = (d / "significance_results.txt").read_text()
    m_ci = re.search(r"Round 0 vs Round 5.*?Bootstrap 95% CI: \[([+\-0-9.]+), ([+\-0-9.]+)\]", sig, re.S)
    m_p = re.search(r"Round 0 vs Round 5.*?Permutation p \(two-sided\): ([0-9.]+)", sig, re.S)
    lines.append(
        f"seed {s}: r0={r0:.4f} best={best_k:.4f}(r{best_r}) r5={r5:.4f} "
        f"delta={r5-r0:+.4f} CI=[{m_ci.group(1)},{m_ci.group(2)}] p={m_p.group(1)}"
    )
lines += ["", f"VERDICT: {verdict}"]
Path("results_bare_all_three_summary.txt").write_text("\n".join(lines) + "\n")
print("Saved results_bare_all_three_summary.txt")
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
