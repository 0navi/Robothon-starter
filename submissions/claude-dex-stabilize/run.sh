#!/usr/bin/env bash
# One-shot reproducer for the canonical metrics + video + multi-seed sweep.
# Usage: bash run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

cd "$ROOT"

python -m pip install -r requirements.txt

echo "=== Canonical 90 s render (seed=12345, force=4 N) ==="
python submissions/claude-dex-stabilize/main.py --seed 12345

echo
echo "=== Multi-seed sweep (10 seeds × 4 N) ==="
python submissions/claude-dex-stabilize/main.py --multi-seed --n 10

echo
echo "=== Difficulty envelope (10 seeds × {2,4,8} N) ==="
python submissions/claude-dex-stabilize/main.py --difficulty-sweep --n 10

echo
echo "=== Unit tests ==="
python -m pytest submissions/claude-dex-stabilize/test_controller.py -v

echo
echo "All done. Outputs in submissions/claude-dex-stabilize/outputs/"
