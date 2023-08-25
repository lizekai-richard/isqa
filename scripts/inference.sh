DATA_PATH="/mnt/data/zekai/processed_scimrc.json"
LORA_PATH="Vicuna-LoRA/checkpoint-1200"
MODEL_PATH="lmsys/vicuna-7b-v1.3"

nohup python3 inference.py \
--data_path $DATA_PATH \
--model_path $MODEL_PATH \
--lora_path $LORA_PATH \
--batch_size 4 \
--num_beams 2 \
--min_new_tokens 100 \
--max_new_tokens 200 \
> logs/inference_vicuna_20230819.log 2>&1 &