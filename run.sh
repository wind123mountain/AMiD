# bash install.sh

# source .venv/bin/activate

bash ./scripts/download_data.sh

if [ ! -d "./processed_data/ultraInteract" ]; then
    bash ./scripts/process_data_ultraInteract.sh
fi

# bash ./scripts/distillm-nnm/train_qwen2.5_1.5B_it_1e.sh
bash ./scripts/distillm-nnm/train_qwen2.5_1.5B_it_2e.sh
# bash ./scripts/distillm-nnm/train_qwen2.5_1.5B_it_3e.sh
# bash ./scripts/distillm-nnm/train_qwen2.5_1.5B_it_1e_2.sh
bash ./scripts/distillm-nnm/train_qwen2.5_1.5B_it_2e_2.sh