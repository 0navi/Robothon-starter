#!/usr/bin/env bash
# One-shot reproducer for the LEAP xylophone submission.
# Usage:  bash submissions/claude-leap-xylophone/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

python -m pip install -r submissions/claude-leap-xylophone/requirements.txt

echo "=== Canonical 48 s render (seed=12345, tempo=110 bpm) ==="
python submissions/claude-leap-xylophone/main.py --seed 12345

echo
echo "=== Multi-seed sweep (10 seeds × 110 bpm) ==="
python submissions/claude-leap-xylophone/main.py --multi-seed --n 10

echo
echo "=== Difficulty sweep (10 seeds × {90, 110, 130} bpm) ==="
python submissions/claude-leap-xylophone/main.py --difficulty-sweep --n 10

echo
echo "=== Unit tests ==="
python -m pytest submissions/claude-leap-xylophone/test_controller.py -v

echo
echo "All done. Outputs in submissions/claude-leap-xylophone/outputs/"
