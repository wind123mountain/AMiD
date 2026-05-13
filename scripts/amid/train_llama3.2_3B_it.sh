#! /bin/bash

GPUS=(0 1 2 3)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=.
CKPT_NAME="llama3.2-3B-Instruct"
CKPT="meta-llama/Llama-3.2-3B-Instruct"
TEACHER_CKPT_NAME="deepSeek-R1-Distill-Llama-8B"
TEACHER_CKPT="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# data
DATA_DIR="${BASE_PATH}/processed_data/ultraInteract/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"
# hp
BATCH_SIZE=4
LR=1e-4
GRAD_ACC=2
EVAL_BATCH_SIZE=32
# length
MAX_LENGTH=1024
# seed
SEED=10

AMID_DIV_NAME="ab"
AMID_DIV_ORDER="pr"
AMID_ALPHA=0.5
AMID_LAM=0.5

SAVE_PATH="${BASE_PATH}/results/${CKPT_NAME}#amid/${AMID_DIV_NAME}_${AMID_DIV_ORDER}_${AMID_ALPHA}_${AMID_LAM}_${BATCH_SIZE}_${LR}"



OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num -1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 3"
OPTS+=" --kd-ratio 1.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 512"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero1_bf16.json" # From MiniLLM to avoid OVERFLOW
# type
OPTS+=" --type adaptive-amid"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# distillm
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
# amid
OPTS+=" --amid-div-name ${AMID_DIV_NAME}"
OPTS+=" --amid-div-order ${AMID_DIV_ORDER}"
OPTS+=" --amid-alpha ${AMID_ALPHA}"
OPTS+=" --amid-lam ${AMID_LAM}"

OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 16"
OPTS+=" --peft-lora-alpha 128"
OPTS+=" --peft-lora-dropout 0.05"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
