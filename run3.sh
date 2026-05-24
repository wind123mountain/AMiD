# bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_2.sh
# bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_3.sh

# bash scripts/distillm-nnm/qwen3/train_qwen3_it_2e.sh
# bash scripts/distillm-nnm/qwen3/train_qwen3_it_2e_2.sh

source .venv/bin/activate

# bash ./scripts/process_data_ultraInteract.sh

# bash scripts/distillm-nnm/gemma2/train_gemma2_it_2e_mae.sh
# bash scripts/distillm-nnm/gemma2/train_gemma2_it_2e_mae_2.sh
bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_mae.sh
bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_mae_2.sh
bash scripts/distillm-nnm/qwen3/train_qwen3_it_2e_mae.sh
bash scripts/distillm-nnm/qwen3/train_qwen3_it_2e_mae_2.sh