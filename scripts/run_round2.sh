#!/usr/bin/env bash
# Round 2: N6 (N3 + 5x200), N7 (N3 + 100 L-BFGS). Same driver pattern as round 1.
set -u
cd "$HOME/git/fusion-simulation"
UV="$HOME/.local/bin/uv"
mkdir -p logs
DRIVER=logs/round2_driver.log

fail() {
    echo "JOB FAILED: $1 $(date -Is)" >>"$DRIVER"
    exit 1
}

echo "=== round2 $(date -Is) ===" >"$DRIVER"

for RUN in N6 N7; do
    echo "START $RUN $(date -Is)" >>"$DRIVER"
    script -qec "$UV run python scripts/run_sweep.py $RUN" "logs/sweep_$RUN.log" </dev/null \
        || fail "$RUN train"
    NAME=$(grep -a SWEEP_RUN_NAME "logs/sweep_$RUN.log" | tail -1 | awk '{print $2}' | tr -d '\r')
    [ -n "$NAME" ] || fail "$RUN run-name missing"
    script -qec "$UV run python -m src.engine.model_evaluation $NAME --plot-quantity residual" \
        "logs/eval_${RUN}_residual.log" </dev/null || fail "$RUN eval residual"
    script -qec "$UV run python -m src.engine.model_evaluation $NAME --plot-quantity flux" \
        "logs/eval_${RUN}_flux.log" </dev/null || fail "$RUN eval flux"
    echo "RUN DONE $RUN -> $NAME $(date -Is)" >>"$DRIVER"
done

echo "ALL DONE $(date -Is)" >>"$DRIVER"
