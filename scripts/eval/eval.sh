#!/bin/bash

hf download VoCuc/AMiD --include "qwen2.5-1.5B-Instruct#amid/ab_pr_0.5_0.5_4_1e-4/7476/*" \
        --local-dir "results"

TP=2

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

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bắt đầu: ${LABEL} ==="

    BASE_ARGS=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --apply_chat_template
        # --fewshot_as_multiturn
        --log_samples
        --output_path "${OUT}"
    )

    # MBPP: không dùng apply_chat_template, không fewshot_as_multiturn
    BASE_ARGS_CODE=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --log_samples
        --output_path "${OUT}"
    )

    # Thêm BASE_ARGS_MATH — không có --fewshot_as_multiturn
    BASE_ARGS_MATH=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --apply_chat_template
        --log_samples
        --output_path "${OUT}"
        --gen_kwargs "max_new_tokens=4096,temperature=0.0"
    )

    {
        echo "=========================================="
        echo "Label: ${LABEL}"
        echo "Start: $(date)"
        echo "=========================================="

        echo ">>> [1/10] GSM8K"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm8k --num_fewshot 5 --gen_kwargs "max_new_tokens=4096"

        # echo ">>> [2/10] MATH (Hendrycks full)"
        # lm_eval "${BASE_ARGS_MATH[@]}" \
        #     --tasks hendrycks_math \
        #     --num_fewshot 4 \
        #     --system_instruction "You are a math teacher. Solve step by step. Put your final answer in \\boxed{ANSWER}."

        echo ">>> [2/10] MATH (Minerva format)"
        lm_eval "${BASE_ARGS[@]}" \
            --tasks minerva_math \
            --num_fewshot 4 --gen_kwargs "max_new_tokens=4096"

        echo ">>> [3/10] MMLU-STEM"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5 --gen_kwargs "max_new_tokens=4096"

        echo ">>> [4/10] SciQ"
        lm_eval "${BASE_ARGS_MATH[@]}" --tasks sciq --num_fewshot 0 --gen_kwargs "max_new_tokens=4096"

        echo ">>> [5/10] MBPP"
        lm_eval "${BASE_ARGS[@]}" --tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --gen_kwargs "max_new_tokens=4096"

        echo ">>> [6/10] GSM-Plus (5-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm_plus --num_fewshot 5 --gen_kwargs "max_new_tokens=4096"

        echo ">>> [7/10] MMLU-Pro-Math (5-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_pro_math --num_fewshot 5 --gen_kwargs "max_new_tokens=4096"

        echo ">>> [8/10] BBH CoT (3-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks bbh_cot_fewshot --num_fewshot 3 --gen_kwargs "max_new_tokens=4096"

        echo ">>> [9/10] MuSR (0-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks leaderboard_musr --num_fewshot 0 --gen_kwargs "max_new_tokens=4096"

        echo ">>> [10/10] IFEval (0-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks leaderboard_ifeval --num_fewshot 0 --gen_kwargs "max_new_tokens=4096"



        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="
    } 2>&1 | tee "${LOG}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}

# CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen3-1.7B#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=Qwen/Qwen3-1.7B,lora_local_path=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0/4984,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.8,trust_remote_code=True"



CUDA_VISIBLE_DEVICES=0,1 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.8,trust_remote_code=True"



CUDA_VISIBLE_DEVICES=0,1 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-it-nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0-1246" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0/1246,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.75,trust_remote_code=True"

echo "=== Done ==="

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.9_K128_L4_epoch2_lr1e-4_kdr0.75/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.9_K128_L4_epoch2_lr1e-4_kdr0.75

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm1.0_K128_L4_epoch2_lr1e-4_kdr0.75/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm1.0_K128_L4_epoch2_lr1e-4_kdr0.75

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0/4984 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0