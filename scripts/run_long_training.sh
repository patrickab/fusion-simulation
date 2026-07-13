#!/usr/bin/env bash
# Does the corrected soft-BC loss beat legacy with a bigger budget?
# Run A: 2400 epochs (4x baseline) — isolates training length.
# Run B: 2400 epochs + 1024 L-BFGS polish steps — the full recipe.
# Ends with ALL DONE / JOB FAILED in logs/driver_long.log.
set -u
cd "$HOME/git/fusion-simulation"
UV="$HOME/.local/bin/uv"
COMMIT=$(git rev-parse --short HEAD)
mkdir -p logs

fail() {
    echo "JOB FAILED: $1" >>logs/driver_long.log
    exit 1
}

echo "=== long-training benchmark $(date) commit=$COMMIT ===" >logs/driver_long.log

echo "--- Run A: 2400 epochs, no lbfgs ---" >>logs/driver_long.log
script -qec "$UV run python -m src.engine.network --soft-bc --fourier-features 0 --huber-delta 0 --lr 2e-4 --epochs 2400" \
    logs/longA.log </dev/null || fail "run A"
RUNA=$(ls -t "data/benchmarks/$COMMIT" | head -1)
echo "run A: $COMMIT/$RUNA" >>logs/driver_long.log

echo "--- Run B: 2400 epochs + lbfgs 1024 ---" >>logs/driver_long.log
script -qec "$UV run python -m src.engine.network --soft-bc --fourier-features 0 --huber-delta 0 --lr 2e-4 --epochs 2400 --lbfgs 1024" \
    logs/longB.log </dev/null || fail "run B"
RUNB=$(ls -t "data/benchmarks/$COMMIT" | head -1)
echo "run B: $COMMIT/$RUNB" >>logs/driver_long.log

for q in residual flux; do
    script -qec "$UV run python -m src.engine.model_evaluation $COMMIT/$RUNA $COMMIT/$RUNB --plot-quantity $q" \
        logs/eval_long_$q.log </dev/null || fail "eval $q"
done
script -qec "$UV run python scripts/psi_stats.py $COMMIT/$RUNA $COMMIT/$RUNB" \
    logs/psi_stats_long.log </dev/null || fail "psi stats"

echo "ALL DONE" >>logs/driver_long.log
