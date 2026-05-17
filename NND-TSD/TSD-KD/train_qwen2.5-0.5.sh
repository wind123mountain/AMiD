beta=0.9
lambda=1.0
threshold=0.1
model_name=Qwen/Qwen2.5-0.5B
teacher_model_name=Qwen/Qwen2.5-Math-1.5B
indirect_kd_alpha=0.1

# NNM (set use_nnm=0 to run pure TSD-KD baseline)
use_nnm=1
nnm_ratio=0.1
nnm_K=128
nnm_n_layers=4
nnm_warmup_steps=200
nnm_ramp_steps=100

if [ "$use_nnm" = "1" ]; then nnm_flag="--nnm"; else nnm_flag="--no-nnm"; fi

accelerate launch --config_file ./accelerate_ddp_config.yaml train.py \
    --beta $beta \
    --lmbda $lambda \
    --threshold $threshold \
    --indirect-kd-alpha $indirect_kd_alpha \
    --student-model $model_name \
    --teacher-model $teacher_model_name \
    $nnm_flag \
    --nnm-ratio $nnm_ratio \
    --nnm-K $nnm_K \
    --nnm-n-layers $nnm_n_layers \
    --nnm-warmup-steps $nnm_warmup_steps \
    --nnm-ramp-steps $nnm_ramp_steps