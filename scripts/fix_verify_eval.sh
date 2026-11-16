#!/usr/bin/env bash
set -euo pipefail
source scripts/env.sh
if [ ! -f metrics/fix_repro_deterministic_warm.json ]; then
  "$NAV3_PYTHON" scripts/fix_repro.py --tag deterministic_warm --warm --deterministic >> logs/fix_repro_deterministic_warm.log 2>&1
fi
for variant in autotune-only ops-only; do
  if [ ! -f "metrics/fix_repro_${variant}.json" ]; then
    "$NAV3_PYTHON" scripts/fix_repro.py --tag "$variant" --"$variant" >> "logs/fix_repro_${variant}.log" 2>&1
  fi
done
for name in ppo_s0 dreamer_s0; do
  if [ ! -f "metrics/fix_repeat_v1_${name}_a.json" ]; then
    "$NAV3_PYTHON" -m nav3.fix_eval --name "$name" --split test --episodes 30 --tag "repeat_v1_${name}_a" >> "logs/fix_repeat_v1_${name}_a.log" 2>&1
  fi
  if [ ! -f "metrics/fix_repeat_v1_${name}_b.json" ]; then
    "$NAV3_PYTHON" -m nav3.fix_eval --name "$name" --split test --episodes 30 --tag "repeat_v1_${name}_b" --compare "metrics/fix_repeat_v1_${name}_a.json" >> "logs/fix_repeat_v1_${name}_b.log" 2>&1
  fi
done
for name in planner dreamer_s0 ppo_s0; do
  for split in heldout test; do
    episodes=5
    if [ "$split" = test ]; then episodes=100; fi
    for ob in 0 1; do
      for sf in 0 1; do
        flags=()
        if [ "$ob" = 1 ]; then flags+=(--ob); fi
        if [ "$sf" = 1 ]; then flags+=(--sf); fi
        if [ ! -f "metrics/fix_eval_v1_${name}_${split}_ob${ob}_sf${sf}.json" ]; then
          "$NAV3_PYTHON" -m nav3.fix_eval --name "$name" --split "$split" --episodes "$episodes" "${flags[@]}" >> "logs/fix_eval_v1_${name}_${split}_ob${ob}_sf${sf}.log" 2>&1
        fi
      done
    done
  done
done
