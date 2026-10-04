# bash install.sh

source .venv/bin/activate

bash ./scripts/download_data.sh

bash ./scripts/process_data_ultraInteract.sh

# if [ ! -d "./processed_data/ultraInteract" ]; then
#     bash ./scripts/process_data_ultraInteract.sh
# fi


CUDA_VISIBLE_DEVICES=0,1 bash scripts/csd/train_gemma2_2B_it.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/amid/train_gemma2_2B_it.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/distillm-nnm/gemma2/train_gemma2_it_2e_2.sh

bash scripts/eval/eval_2.sh