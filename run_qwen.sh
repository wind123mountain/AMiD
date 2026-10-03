#!/bin/bash
# Qwen2.5-0.5B-Instruct <- Qwen3-4B-Instruct-2507 on 8×H200: CSD -> AMiD -> DistiLLM+NNM, one 8-GPU DDP phase at a time.
# B_global = 8 GPU × 8 per device × 1 grad_acc = 64. The 2-GPU setup used 2 × 16 × 2; 16 per device
# cannot give 64 on 8 GPUs, so the per-device batch drops to 8 and the global batch stays 64.
# Launch:  setsid nohup bash run_qwen.sh > /dev/null 2>&1 < /dev/null &
# Logs:    runs/$RUN_ID/qwen/{driver.log,<phase>.log,<phase>.exit_code,run.pid,exit_code}
# bash install.sh

cd "$(dirname "$0")"
source scripts/run_8xh200_common.sh
init_run qwen

source .venv/bin/activate

hf_prepare "${PINNED_MODELS[@]}" || exit 1
prepare_data Qwen/Qwen3-4B-Instruct-2507 || exit 1
# Training stays online: with HF_HUB_OFFLINE=1, transformers 4.57.3 still calls the Hub when loading
# any tokenizer with vocab > 100k from a repo id (_patch_mistral_regex), so get_tokenizer would fail.
# run_phase checks after every phase that the cached `main` refs still equal the pinned SHAs.
export BATCH_SIZE=8

run_phase csd scripts/csd/train_qwen2.5_0.5B_it.sh "./results/qwen2.5-0.5B-it#csd/ab_pr_0.5_0.5_8_1e-4" || exit $?
run_phase amid scripts/amid/train_qwen2.5_0.5B_it.sh "./results/qwen2.5-0.5B-it#amid/ab_pr_0.5_0.5_8_1e-4" || exit $?
run_phase nnm scripts/distillm-nnm/qwen2.5/train_qwen2.5_0.5B_it_2e.sh "./results/qwen2.5-0.5B-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" || exit $?
log "all phases done"
