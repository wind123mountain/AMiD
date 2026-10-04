#!/bin/bash
# lm-eval benchmarks (scripts/eval/eval_2.sh) for the 5 final checkpoints of run_gemma.sh + run_qwen.sh,
# on all 8 H200 (vLLM, data_parallel_size=8 through ray). Runs only after every training phase exited 0.
# Launch:  setsid nohup bash run_eval.sh > /dev/null 2>&1 < /dev/null &
# Logs:    runs/$RUN_ID/eval/{driver.log,eval_2.log,eval_2.exit_code,run.pid,exit_code}

cd "$(dirname "$0")"
source scripts/run_8xh200_common.sh
init_run eval

source .venv/bin/activate

for ph in gemma/amid gemma/nnm qwen/csd qwen/amid qwen/nnm; do
    if [ "$(cat "runs/$RUN_ID/$ph.exit_code" 2>/dev/null)" != 0 ] || [ ! -s "runs/$RUN_ID/$ph.ckpt" ]; then
        log "eval: $ph has no exit code 0 or no checkpoint record, not evaluating"
        exit 1
    fi
done

plog="$RUN_DIR/eval_2.log"
[ -f "$plog" ] && mv "$plog" "$plog.$(date -u +%Y%m%dT%H%M%SZ)"
rm -f "$RUN_DIR/eval_2.exit_code"
require_idle_gpus
log "eval: start scripts/eval/eval_2.sh"
echo eval_2 > "$RUN_DIR/current_phase"
bash scripts/eval/eval_2.sh > "$plog" 2>&1
rc=$?
echo $rc > "$RUN_DIR/eval_2.exit_code"
log "eval: exit $rc"
exit $rc
