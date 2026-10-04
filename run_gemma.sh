#!/bin/bash
# gemma-2-2b-it <- gemma-2-9b-it on 8×H200: CSD -> AMiD -> DistiLLM+NNM, one 8-GPU DDP phase at a time.
# B_global = 64: csd/amid 8 GPU × 4 per device × 2 grad_acc, nnm 8 × 2 × 4 (the scripts derive grad_acc).
# Checkpoints go to results/$RUN_ID/..., so earlier runs' results/gemma2-2b-it#*/ dirs are never touched.
# Launch:  setsid nohup bash -c 'bash run_gemma.sh && bash run_eval.sh' > /dev/null 2>&1 < /dev/null &
# Logs:    runs/$RUN_ID/gemma/{driver.log,<phase>.log,<phase>.exit_code,<phase>.ckpt,run.pid,exit_code}
# bash install.sh

cd "$(dirname "$0")"
export RUN_ID=${RUN_ID:-gemma-kd-len2048}
source scripts/run_8xh200_common.sh
# Gemma pair only: a moved Qwen `main` must not stop this run.
PINNED_MODELS=("${PINNED_MODELS[@]:0:2}")
init_run gemma

source .venv/bin/activate

hf_prepare "${PINNED_MODELS[@]}" || exit 1
prepare_data google/gemma-2-9b-it || exit 1
# Training stays online: with HF_HUB_OFFLINE=1, transformers 4.57.3 still calls the Hub when loading
# any tokenizer with vocab > 100k from a repo id (_patch_mistral_regex), so get_tokenizer would fail.
# run_phase checks after every phase that the cached `main` refs still equal the pinned SHAs.

R="./results/$RUN_ID"
run_phase csd scripts/csd/train_gemma2_2B_it.sh "$R/gemma2-2b-it#csd/csd_ab_pr_0.5_0.5_4_1e-4" || exit $?
run_phase amid scripts/amid/train_gemma2_2B_it.sh "$R/gemma2-2b-it#amid/ab_pr_0.5_0.5_4_1e-4" || exit $?
run_phase nnm scripts/distillm-nnm/gemma2/train_gemma2_it_2e_2.sh "$R/gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" || exit $?
log "all phases done"
