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

    if [ -f "${OUT}/DONE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Bỏ qua (đã xong): ${LABEL} ==="
        return 0
    fi
    mkdir -p "${OUT}"
    mkdir -p "$(dirname "${LOG}")"
    rm -f "${OUT}/FAILED"

    # lm_eval with its exit code recorded in ${OUT}/FAILED (the block below runs in a pipe subshell)
    task() {
        lm_eval "$@"
        local rc=$?
        [ $rc -eq 0 ] || { echo "!! FAILED rc=${rc}: lm_eval $*"; echo "rc=${rc} $*" >> "${OUT}/FAILED"; }
    }

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
        task "${BASE_ARGS[@]}" --tasks gsm8k

        echo ">>> [2/10] MATH (Minerva format)"
        task "${BASE_ARGS[@]}" \
            --tasks minerva_math \
            --num_fewshot 4

        echo ">>> [3/10] MMLU-STEM"
        task "${BASE_ARGS[@]}" --tasks mmlu_stem --num_fewshot 5

        echo ">>> [4/10] SciQ"
        task "${BASE_ARGS[@]}" --tasks sciq

        echo ">>> [5/10] MBPP"
        task "${BASE_ARGS_CODE[@]}" --tasks mbpp --confirm_run_unsafe_code  --num_fewshot 3

        echo ">>> [6/10] GSM-Plus (5-shot)"
        task "${BASE_ARGS[@]}" --tasks gsm_plus


        echo "=========================================="
        echo "DONE: ${LABEL} | $(date)"
        echo "=========================================="
    } 2>&1 | tee "${LOG}"

    if [ -f "${OUT}/FAILED" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === LỖI: ${LABEL} ($(wc -l < "${OUT}/FAILED") task) ==="
        return 1
    fi
    touch "${OUT}/DONE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Xong: ${LABEL} ==="
}

# Final checkpoints of run_gemma.sh / run_qwen.sh, read from the <phase>.ckpt records written by
# check_ckpt (scripts/run_8xh200_common.sh), so only validated checkpoints are evaluated.
CKPT_RECORDS="runs/${RUN_ID:-amid-kd-8xh200}"
final_ckpt() {
    local c
    c=$(sed -n 's/^final_checkpoint=\([^ ]*\) .*/\1/p' "${CKPT_RECORDS}/$1.ckpt" 2>/dev/null)
    [ -d "$c" ] || { echo "no final checkpoint for $1 in ${CKPT_RECORDS}/$1.ckpt" >&2; exit 1; }
    echo "$c"
}

# Full fine-tune checkpoints are saved with model.module.save_pretrained, so the NNM phase also stores
# its projectors.* weights, which vLLM rejects. Such a checkpoint gets a copy without them in
# outputs/vllm_ckpt/<label>; checkpoints without projectors are used in place.
vllm_ckpt() {
    python - "$1" "outputs/vllm_ckpt/$2" <<'PY' || exit 1
import os, shutil, sys, torch
src, dst = sys.argv[1:]
bins = sorted(f for f in os.listdir(src) if f.startswith("pytorch_model") and f.endswith(".bin"))
if bins != ["pytorch_model.bin"]:
    sys.exit(f"{src}: expected one pytorch_model.bin, found {bins}")
sd = torch.load(os.path.join(src, "pytorch_model.bin"), map_location="cpu", weights_only=True)
drop = [k for k in sd if k.startswith("projectors.")]
if not drop:
    print(src)
    sys.exit()
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if not f.startswith("pytorch_model") and os.path.isfile(os.path.join(src, f)):
        shutil.copy2(os.path.join(src, f), dst)
torch.save({k: v for k, v in sd.items() if k not in drop}, os.path.join(dst, "pytorch_model.bin"))
print(f"dropped {len(drop)} projector tensors: {src} -> {dst}", file=sys.stderr)
print(dst)
PY
}

GEMMA_REV=299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8
GEMMA_AMID=$(final_ckpt gemma/amid) || exit 1
GEMMA_NNM=$(final_ckpt gemma/nnm) || exit 1
QWEN_CSD=$(vllm_ckpt "$(final_ckpt qwen/csd)" "qwen2.5-0.5B-it#csd") || exit 1
QWEN_AMID=$(vllm_ckpt "$(final_ckpt qwen/amid)" "qwen2.5-0.5B-it#amid") || exit 1
QWEN_NNM=$(vllm_ckpt "$(final_ckpt qwen/nnm)" "qwen2.5-0.5B-it#sfkl_nnm") || exit 1
FAILED_ANY=0


HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma2-2b-it#amid/ab_pr_0.5_0.5_4_1e-4" \
    "pretrained=google/gemma-2-2b-it,revision=${GEMMA_REV},lora_local_path=${GEMMA_AMID},data_parallel_size=${DP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True" || FAILED_ANY=1


HF_ALLOW_CODE_EVAL=1 run_eval \
    "gemma2-2b-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=google/gemma-2-2b-it,revision=${GEMMA_REV},lora_local_path=${GEMMA_NNM},data_parallel_size=${DP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True,max_lora_rank=32,enable_lora=True" || FAILED_ANY=1


HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-0.5B-it#csd/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=${QWEN_CSD},data_parallel_size=${DP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True" || FAILED_ANY=1


HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-0.5B-it#amid/ab_pr_0.5_0.5_8_1e-4" \
    "pretrained=${QWEN_AMID},data_parallel_size=${DP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True" || FAILED_ANY=1


HF_ALLOW_CODE_EVAL=1 run_eval \
    "qwen2.5-0.5B-it#sfkl_nnm_lora/nnm0.2_K128_L4_epoch2_lr1e-4_kdr1.0" \
    "pretrained=${QWEN_NNM},data_parallel_size=${DP},dtype=bfloat16,gpu_memory_utilization=0.8,trust_remote_code=True" || FAILED_ANY=1


echo "Eval Done! (failed models: ${FAILED_ANY})"
exit ${FAILED_ANY}
