#!/bin/bash

hf download VoCuc/nnm --include "qwen2.5-1.5B-Instruct#amid/ab_pr_0.5_0.5_4_1e-4/7476/*" \
        --local-dir "results"

hf download VoCuc/nnm-eval --include "layer_analysis/6_method_tsd/hidden_states/*" \
        --local-dir "results"

TP=4

LOG_DIR="outputs/eval_results_2/logs"
OUT_DIR="outputs/eval_results_2/vllm"
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

        # echo ">>> [1/10] GSM8K"
        # lm_eval "${BASE_ARGS[@]}" --tasks gsm8k

        # echo ">>> [2/10] MATH (Hendrycks full)"
        # lm_eval "${BASE_ARGS_MATH[@]}" \
        #     --tasks hendrycks_math \
        #     --num_fewshot 4 \
            
        # echo ">>> [2/10] MATH (Minerva format)"
        # lm_eval "${BASE_ARGS[@]}" \
        #     --tasks minerva_math \
            # --num_fewshot 4

        echo ">>> [3/10] MMLU-STEM"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        # echo ">>> [4/10] SciQ"
        # lm_eval "${BASE_ARGS[@]}" --tasks sciq

        # echo ">>> [5/10] MBPP"
        # lm_eval "${BASE_ARGS_CODE[@]}" --tasks mbpp --confirm_run_unsafe_code  --num_fewshot 3

        # echo ">>> [5/10] MBPP"
        # lm_eval "${BASE_ARGS_CODE[@]}" --tasks mbpp_instruct --confirm_run_unsafe_code  --num_fewshot 3

        # echo ">>> [6/10] GSM-Plus (5-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks gsm_plus

        # echo ">>> [7/10] MMLU-Pro-Math (5-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks mmlu_pro_math

        # echo ">>> [8/10] BBH CoT (3-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks bbh_cot_fewshot

        # echo ">>> [9/10] MuSR (0-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks leaderboard_musr --num_fewshot 0

        # echo ">>> [10/10] IFEval (0-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks leaderboard_ifeval

        # echo ">>> [10/10] IFEval (0-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks ifeval


        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="
    } 2>&1 | tee "${LOG}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}


run_eval_no_chat() {
    local LABEL=$1
    local MODEL_ARGS=$2
    local OUT="${OUT_DIR}/no_chat_${LABEL}"
    local LOG="${LOG_DIR}/no_chat_${LABEL}.log"

    mkdir -p "${OUT}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bắt đầu: ${LABEL} ==="

    BASE_ARGS=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        # --fewshot_as_multiturn
        --log_samples
        --output_path "${OUT}"
        --gen_kwargs "max_new_tokens=4096"
    )

    # MBPP: không dùng apply_chat_template, không fewshot_as_multiturn
    BASE_ARGS_CODE=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
        --log_samples
        --output_path "${OUT}"
        --apply_chat_template
        --gen_kwargs "max_new_tokens=4096,temperature=0.0"
    )

    # Thêm BASE_ARGS_MATH — không có --fewshot_as_multiturn
    BASE_ARGS_MATH=(
        --model vllm
        --model_args "${MODEL_ARGS}"
        --batch_size auto
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
        lm_eval "${BASE_ARGS[@]}" --tasks gsm8k

        echo ">>> [2/10] MATH (Hendrycks full)"
        lm_eval "${BASE_ARGS_MATH[@]}" \
            --tasks hendrycks_math \
            --num_fewshot 4 \
            
        echo ">>> [2/10] MATH (Minerva format)"
        lm_eval "${BASE_ARGS[@]}" \
            --tasks minerva_math \
            # --num_fewshot 4

        echo ">>> [3/10] MMLU-STEM"
        lm_eval "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        # echo ">>> [4/10] SciQ"
        # lm_eval "${BASE_ARGS[@]}" --tasks sciq

        echo ">>> [5/10] MBPP"
        lm_eval "${BASE_ARGS_CODE[@]}" --tasks mbpp --confirm_run_unsafe_code  --num_fewshot 3

        echo ">>> [5/10] MBPP"
        lm_eval "${BASE_ARGS_CODE[@]}" --tasks mbpp_instruct --confirm_run_unsafe_code  --num_fewshot 3

        echo ">>> [6/10] GSM-Plus (5-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks gsm_plus

        # echo ">>> [7/10] MMLU-Pro-Math (5-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks mmlu_pro_math

        # echo ">>> [8/10] BBH CoT (3-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks bbh_cot_fewshot

        # echo ">>> [9/10] MuSR (0-shot)"
        # lm_eval "${BASE_ARGS[@]}" --tasks leaderboard_musr --num_fewshot 0

        echo ">>> [10/10] IFEval (0-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks leaderboard_ifeval

        echo ">>> [10/10] IFEval (0-shot)"
        lm_eval "${BASE_ARGS[@]}" --tasks ifeval


        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="
    } 2>&1 | tee "${LOG}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L2_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L2_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L8_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L8_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"


CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_3_0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_3_0.1_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_4_0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_4_0.1_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"


CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#csd/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=results/qwen3-1.7B#csd/ab_pr_0.5_0.5_8_1e-4,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,enable_thinking=False"


CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#amid/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=results/qwen3-1.7B#amid/ab_pr_0.5_0.5_8_1e-4,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,enable_thinking=False"



CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora_mae/nnm0.1_K128_L4_epoch2_lr5e-5_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora_mae/nnm0.1_K128_L4_epoch2_lr5e-5_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"



CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it#amid/ab_pr_0.5_0.5_2_1e-4" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#csd_csd/gemma2-2b-it#csd/ab_pr_0.5_0.5_2_1e-4/7476,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it#csd/ab_pr_0.5_0.5_2_1e-4" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#csd_csd/gemma2-2b-it#csd/csd_ab_pr_0.5_0.5_4_1e-4/7476,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"



CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#csd/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=results/qwen3-1.7B#csd/ab_pr_0.5_0.5_8_1e-4,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,enable_thinking=False"


CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#amid/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=results/qwen3-1.7B#amid/ab_pr_0.5_0.5_8_1e-4,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,enable_thinking=False"



CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora_mae/nnm0.1_K128_L4_epoch2_lr5e-5_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora_mae/nnm0.1_K128_L4_epoch2_lr5e-5_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"


# CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,reasoning_parser=deepseek_r1,enable_thinking=True,think_end_token=\"</think>\""


CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,enable_thinking=False"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr0.75" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr0.75,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm0.7_K128_L4_epoch2_lr1e-4_kdr0.75" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.7_K128_L4_epoch2_lr1e-4_kdr0.75,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm0.9_K128_L4_epoch2_lr1e-4_kdr0.75" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.9_K128_L4_epoch2_lr1e-4_kdr0.75,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm1.0_K128_L4_epoch1_lr1e-4_kdr1.0" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm1.0_K128_L4_epoch1_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm_new0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm_new0.1_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#sfkl_nnm_lora/nnm_new0.3_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen3-1.7B#sfkl_nnm_lora/nnm_new0.3_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.1_K128_L4_epoch2_lr1e-4_kdr0.9" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr0.9/4984,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True,use_fast=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0/4984,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.3_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.3_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.1_K128_L4_epoch2_lr1e-4_kdr0.9_ckpt2492" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr0.9/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0_ckpt1246" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0/1246,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0_ckpt2492" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.3_K128_L4_epoch2_lr1e-4_kdr1.0_ckpt1246" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.3_K128_L4_epoch2_lr1e-4_kdr1.0/1246,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma-2-2b-it/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0_ckpt1246" \
    "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0/1246,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#amid/ab_pr_0.5_0.5_4_1e-4" \
    "pretrained=Qwen/Qwen3-1.7B,lora_local_path=results/qwen3-1.7B#amid/ab_pr_0.5_0.5_4_1e-4/7476,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#amid/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=Qwen/Qwen3-1.7B,lora_local_path=results/qwen3-1.7B#amid/ab_pr_0.5_0.5_8_1e-4/7476,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen3-1.7B#csd/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=Qwen/Qwen3-1.7B,lora_local_path=results/qwen3-1.7B#csd/ab_pr_0.5_0.5_8_1e-4/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K256_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K256_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K32_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K32_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K64_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.2_K64_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.3_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.3_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.7_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.7_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.9_K128_L4_epoch2_lr1e-4_kdr0.75" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.9_K128_L4_epoch2_lr1e-4_kdr0.75,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm1.0_K128_L4_epoch2_lr1e-4_kdr0.75" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm1.0_K128_L4_epoch2_lr1e-4_kdr0.75,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.2_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"

CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.5_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.5_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True"


# CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "gemma-2-2b-it/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.2,trust_remote_code=True,max_lora_rank=32,enable_lora=True"




# CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=Qwen/Qwen3-1.7B,lora_local_path=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0/4984,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32"


# CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen3-1.7" \
#     "pretrained=Qwen/Qwen3-1.7B,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,reasoning_parser=deepseek_r1,enable_thinking=True,think_end_token=\"</think>\""


# CUDA_VISIBLE_DEVICES=0,1 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/erank0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/erank0.2_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.8,trust_remote_code=True"


# CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "gemma-2-2b-it/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.7,trust_remote_code=True,max_lora_rank=32,enable_lora=True"



# CUDA_VISIBLE_DEVICES=2,3 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.7_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.7_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.8,trust_remote_code=True"


# CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen2.5-1.5b-tsd" \
#     "pretrained=Minsang/TSD-KD_Qwen2.5-1.5B,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.7,trust_remote_code=True"



# CUDA_VISIBLE_DEVICES=6,7 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen3-1.7" \
#     "pretrained=Qwen/Qwen3-1.7B,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,reasoning_parser=deepseek_r1,enable_thinking=True,think_end_token=\"</think>\""

# CUDA_VISIBLE_DEVICES=2,3 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "gemma-2-2b-it/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.7,trust_remote_code=True,max_lora_rank=32"



# CUDA_VISIBLE_DEVICES=2,3 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen2.5-1.5B-Instruct#sfkl_nnm_lora/erank0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/erank0.2_K128_L4_epoch2_lr1e-4_kdr1.0,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.8,trust_remote_code=True"


# CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "gemma-2-2b-it/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=google/gemma-2-2b-it,lora_local_path=results/gemma2-2b-it#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.7,trust_remote_code=True"



# CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen3-1.7B#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=Qwen/Qwen3-1.7B,lora_local_path=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0/4984,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.8,trust_remote_code=True"



# CUDA_VISIBLE_DEVICES=4,5 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen3-1.7B#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0" \
#     "pretrained=Qwen/Qwen3-1.7B,lora_local_path=results/qwen3-1.7B#sfkl_nnm_lora/nnm0.5_K128_L4_epoch2_lr1e-4_kdr1.0/4984,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.8,trust_remote_code=True"



# CUDA_VISIBLE_DEVICES=0,1 HF_ALLOW_CODE_EVAL=1 run_eval \
#     "qwen2.5-1.5B-it-nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0-1246" \
#     "pretrained=results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch1_lr1e-4_kdr1.0/1246,tensor_parallel_size=${TP},dtype=float16,gpu_memory_utilization=0.75,trust_remote_code=True"

echo "=== Done ==="

# python tools/merge_model.py \
#   --base_model Qwen/Qwen2.5-1.5B-Instruct \
#   --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.7_K128_L4_epoch2_lr1e-4_kdr1.0/2492 \
#   --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.7_K128_L4_epoch2_lr1e-4_kdr1.0/2492

# python tools/merge_model.py \
#   --base_model Qwen/Qwen2.5-1.5B-Instruct \
#   --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm1.0_K128_L4_epoch2_lr1e-4_kdr0.75/2492 \
#   --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm1.0_K128_L4_epoch2_lr1e-4_kdr0.75

# python tools/merge_model.py \
#   --base_model Qwen/Qwen2.5-1.5B-Instruct \
#   --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492 \
#   --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm0.1_K128_L4_epoch2_lr1e-4_kdr1.0

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_2_0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_2_0.1_K128_L4_epoch2_lr1e-4_kdr1.0

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_3_0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_3_0.1_K128_L4_epoch2_lr1e-4_kdr1.0

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_4_0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new_finetune_4_0.1_K128_L4_epoch2_lr1e-4_kdr1.0


python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L4_epoch2_lr1e-4_kdr1.0/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L4_epoch2_lr1e-4_kdr1.0

python tools/merge_model.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L8_epoch2_lr1e-4_kdr1.0/2492 \
  --output results/qwen2.5-1.5B-Instruct#sfkl_nnm_lora/nnm_new0.1_K128_L8_epoch2_lr1e-4_kdr1.0

python tools/merge_model.py \
  --base_model Qwen/Qwen3-1.7B \
  --adapter results/qwen3-1.7B#amid/ab_pr_0.5_0.5_8_1e-4/7476 \
  --output results/qwen3-1.7B#amid/ab_pr_0.5_0.5_8_1e-4