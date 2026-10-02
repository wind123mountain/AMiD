# bash install.sh

source .venv/bin/activate

bash ./scripts/download_data.sh

if [ ! -d "./processed_data/ultraInteract" ]; then
    bash ./scripts/process_data_ultraInteract.sh
fi


CUDA_VISIBLE_DEVICES=0,1 bash scripts/csd/train_qwen2.5_0.5B_it.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/amid/train_qwen2.5_0.5B_it.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/distillm-nnm/qwen2.5/train_qwen2.5_0.5B_it_2e.sh