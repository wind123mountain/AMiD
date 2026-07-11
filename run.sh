bash install.sh

source .venv/bin/activate

bash ./scripts/download_data.sh

if [ ! -d "./processed_data/ultraInteract" ]; then
    bash ./scripts/process_data_ultraInteract.sh
fi


bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_1.5B_it_2e_noproj.sh
bash scripts/feature/train_qwen2.5_1.5B_it.sh

bash scripts/eval/eval_2.sh