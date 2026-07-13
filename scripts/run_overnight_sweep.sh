#!/usr/bin/env bash
# Overnight benchmark sweep (benchmark-plan.md): R1-R5 legacy-family + N1-N3
# hard-BC family, each followed by the fixed benchmark eval (residual + flux).
# Driver log: logs/sweep_driver.log — one "RUN DONE ..." line per finished run,
# ends with ALL DONE / JOB FAILED.
set -u
cd "$HOME/git/fusion-simulation"
UV="$HOME/.local/bin/uv"
mkdir -p logs
DRIVER=logs/sweep_driver.log

fail() {
    echo "JOB FAILED: $1 $(date -Is)" >>"$DRIVER"
    exit 1
}

echo "=== overnight sweep $(date -Is) commit=$(git rev-parse --short HEAD) ===" >"$DRIVER"

# front-loaded: sanity, budget baselines, architecture, batch — then the rest,
# so a short night still answers the core soft-vs-hard + size + batch questions.
for RUN in R1 R2 N1 R6 N4 R7 N5 R3 R4 R5 N2 N3; do
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
