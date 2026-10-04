#!/bin/bash
# CPU-only preflight for run_gemma.sh / run_qwen.sh: HF auth, pinned model prefetch, pinned dataset
# download and tokenisation for both teachers. Output: runs/$RUN_ID/preflight/.
cd "$(dirname "$0")/.."
source scripts/run_8xh200_common.sh
RUN_DIR="runs/${RUN_ID}/preflight"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/preflight.log") 2>&1
source .venv/bin/activate

log "preflight start"
hf_prepare "${PINNED_MODELS[@]}" || exit 1
for teacher in google/gemma-2-9b-it Qwen/Qwen3-4B-Instruct-2507; do
    mkdir -p "$RUN_DIR/$teacher"
    RUN_DIR="runs/${RUN_ID}/preflight/$teacher" prepare_data "$teacher" || exit 1
done
log "preflight ok"
