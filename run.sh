# bash install.sh

# source .venv/bin/activate

# bash ./scripts/download_data.sh

# if [ ! -d "./processed_data/ultraInteract" ]; then
#     bash ./scripts/process_data_ultraInteract.sh
# fi

# bash ./scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e.sh
# bash ./scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_2.sh
# bash ./scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_3.sh

# bash scripts/distillm-nnm/qwen3/train_qwen3_it_1e_2.sh
# bash scripts/distillm-nnm/qwen3/train_qwen3_it_1e.sh
# bash scripts/distillm-nnm/qwen3/train_qwen3_it_1e_3.sh
# bash scripts/distillm-nnm/qwen3/train_qwen3_it_2e.sh
# bash scripts/distillm-nnm/qwen3/train_qwen3_it_2e_2.sh

bash scripts/distillm-nnm/gemma2/train_gemma2_it_2e.sh
bash scripts/distillm-nnm/gemma2/train_gemma2_it_2e_2.sh
bash scripts/distillm-nnm/gemma2/train_gemma2_it_2e_3.sh

bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_2.sh
bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_3.sh

bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_4.sh
bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_5.sh
bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_6.sh

bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_1e_ablation.sh bnm
bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_1e_ablation.sh bnmm
bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_1e_ablation.sh erank
bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_1e_with_layer_weighting.sh nnm
