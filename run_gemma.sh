bash install.sh

source .venv/bin/activate

bash ./scripts/download_data.sh

if [ ! -d "./processed_data/ultraInteract" ]; then
    bash ./scripts/process_data_ultraInteract.sh
fi


bash scripts/csd/train_gemma2_2B_it.sh
bash scripts/amid/train_gemma2_2B_it.sh