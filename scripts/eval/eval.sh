#!/bin/bash


TP=2

LOG_DIR="outputs/eval_results/logs"
OUT_DIR="outputs/eval_results/vllm"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"


run_eval() {
    local LABEL=$1
    local MODEL_ARGS=$2
    local OUT="${OUT_DIR}/${LABEL}"
    local LOG="${LOG_DIR}/${LABEL}.log"

    mkdir -p "${OUT}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bắt đầu: ${LABEL} ==="

    BASE_ARGS=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --apply_chat_template
        --fewshot_as_multiturn
        --log_samples
        --output_path "${OUT}"
    )

    {
        echo "=========================================="
        echo "Label: ${LABEL}"
        echo "Start: $(date)"
        echo "=========================================="

        echo ">>> [1/10] GSM8K (CoT)"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm8k --num_fewshot 5

        echo ">>> [2/10] MATH500 (tốt nhất)"
        lm_eval "${BASE_ARGS[@]}" --tasks hendrycks_math500 --num_fewshot 4

        echo ">>> [3/10] MMLU-STEM"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> [4/10] SciQ"
        lm_eval "${BASE_ARGS[@]}" --tasks sciq --num_fewshot 0

        echo ">>> [5/10] MBPP"
        lm_eval "${BASE_ARGS[@]}" --tasks mbpp --num_fewshot 0 --confirm_run_unsafe_code

        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="
    } 2>&1 | tee "${LOG}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}



CUDA_VISIBLE_DEVICES=0,1 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-it-nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0-1246" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0/1246,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.75,trust_remote_code=True"


echo "=== Done ==="

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0/1246 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0/1246
