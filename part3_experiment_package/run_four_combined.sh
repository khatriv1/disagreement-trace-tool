#!/usr/bin/env bash
# Run all four combined signal conditions sequentially.
set -euo pipefail
cd "$(dirname "$0")"
LOG="four_conditions_run.log"
CONDITIONS=(label_reasoning label_ambiguity reasoning_ambiguity all_three)

if [[ -d ../.venv ]]; then
  export PATH="../.venv/bin:$PATH"
elif [[ -d .venv ]]; then
  export PATH=".venv/bin:$PATH"
fi

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "PART 3 — four combined conditions sequential run"
echo "Started: $(date)"
echo "============================================================"

for i in "${!CONDITIONS[@]}"; do
  COND="${CONDITIONS[$i]}"
  N=$((i + 1))
  echo ""
  echo "############################################################"
  echo "# CONDITION $N/4: $COND — started $(date)"
  echo "############################################################"
  ./run_experiment.sh "$COND"
  echo ""
  echo "# CONDITION $N/4: $COND — FINISHED $(date)"
  echo "# Zip: results_${COND}.zip"
  ls -la "results_${COND}.zip" 2>/dev/null || echo "WARNING: zip missing"
done

echo ""
echo "============================================================"
echo "ALL FOUR CONDITIONS COMPLETE — $(date)"
echo "============================================================"

python3 << 'PYEOF'
import csv
from pathlib import Path

conds = ["label_reasoning", "label_ambiguity", "reasoning_ambiguity", "all_three"]
print(f"\n{'condition':<22} {'r0_kappa':>10} {'best_kappa':>10} {'best_r':>7} {'final_kappa':>11}")
print("-" * 65)
for c in conds:
    p = Path("results") / c / "results.csv"
    if not p.exists():
        print(f"{c:<22} {'MISSING':>10}")
        continue
    rows = list(csv.DictReader(p.open()))
    r0 = float(rows[0]["pooled_kappa"])
    best_r, best_k = 0, -1.0
    for row in rows:
        r = int(row["round"])
        if r == 0:
            continue
        k = float(row["pooled_kappa"])
        if k > best_k:
            best_k, best_r = k, r
    final = float(rows[-1]["pooled_kappa"])
    print(f"{c:<22} {r0:>10.4f} {best_k:>10.4f} {best_r:>7} {final:>11.4f}")

print("\nZip files:")
for c in conds:
    z = Path(f"results_{c}.zip")
    if z.exists():
        print(f"  OK  {z.name} ({z.stat().st_size // 1024} KB)")
    else:
        print(f"  MISSING  {z.name}")
PYEOF

echo ""
echo "Done."
