#!/bin/bash


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
DP=${#GPUS[@]}

LOG_DIR="outputs/eval_results/logs"
OUT_DIR="outputs/eval_results/vllm"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# Auto-patch bug hendrycks_math: Answer is not a string
TASK_FILE=$(python -c "import lm_eval, os; print(os.path.join(os.path.dirname(lm_eval.__file__), 'api/task.py'))" 2>/dev/null)
if [ -n "${TASK_FILE}" ]; then
    if ! grep -q "if not isinstance(answer_text, str): answer_text = str(answer_text)" "${TASK_FILE}"; then
        sed -i 's/assert isinstance(answer_text, str).*/if not isinstance(answer_text, str): answer_text = str(answer_text)/' "${TASK_FILE}"
        echo "[PATCH] Patched hendrycks_math bug: ${TASK_FILE}"
    fi
fi

run_eval() {
    local LABEL=$1
    local MODEL_ARGS=$2
    local OUT="${OUT_DIR}/${LABEL}"
    local LOG="${LOG_DIR}/${LABEL}.log"

    mkdir -p "${OUT}"
    mkdir -p "$(dirname "${LOG}")"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bắt đầu: ${LABEL} ==="

    BASE_ARGS=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --apply_chat_template
        --fewshot_as_multiturn
        --log_samples
        --output_path "${OUT}"
        --gen_kwargs "max_new_tokens=5120"
    )

    # MBPP: không dùng apply_chat_template, không fewshot_as_multiturn
    BASE_ARGS_CODE=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --log_samples
        --output_path "${OUT}"
        # --apply_chat_template
        --gen_kwargs "max_new_tokens=5120,temperature=0.0"
    )

    # Thêm BASE_ARGS_MATH — không có --fewshot_as_multiturn
    BASE_ARGS_MATH=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --apply_chat_template
        # --fewshot_as_multiturn
        --log_samples
        --output_path "${OUT}"
        --gen_kwargs "max_new_tokens=5120,temperature=0.0"
    )

    {
        echo "=========================================="
        echo "Label: ${LABEL}"
        echo "Start: $(date)"
        echo "=========================================="

        echo ">>> [1/10] GSM8K"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm8k

        echo ">>> [2/10] MATH (Minerva format)"
        lm_eval "${BASE_ARGS[@]}" \
            --tasks minerva_math \
            --num_fewshot 4

        echo ">>> [3/10] MMLU-STEM"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> [4/10] SciQ"
        lm_eval "${BASE_ARGS[@]}" --tasks sciq

        echo ">>> [5/10] MBPP"
        lm_eval "${BASE_ARGS_CODE[@]}" --tasks mbpp --confirm_run_unsafe_code  --num_fewshot 3

        echo ">>> [6/10] GSM-Plus (5-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm_plus


        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="
    } 2>&1 | tee "${LOG}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}

# bash scripts/eval/merge_lora.sh "results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_no_train_proj"
# bash scripts/eval/merge_lora.sh "results/qwen2.5-1.5B-Instruct#feature"


HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it#csd/csd_ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=./results/gemma2-2b-it#csd/ab_pr_0.5_0.5_8_1e-4/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"


HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it#amid/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=./results/gemma2-2b-it#amid/ab_pr_0.5_0.5_8_1e-4/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"


HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=./results/gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"


HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-0.5-it#csd/csd_ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=results/qwen2.5-0.5-it#csd/ab_pr_0.5_0.5_8_1e-4/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"


HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-0.5-it/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=results/qwen2.5-0.5-it#amid/ab_pr_0.5_0.5_8_1e-4/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"


HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-0.5-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-0.5-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"


echo "Eval Done!"