DATA_PATH="../data/scimrc_for_inst_tune_sec_based.json"
LORA_PATH="Vicuna-LoRA/checkpoint-1200"
MODEL_PATH="../llm_weights/vicuna-7b-v1.3"

nohup python3 evaluate.py \
--data_path $DATA_PATH \
--model_path $MODEL_PATH \
--lora_path $LORA_PATH \
--batch_size 4 \
--num_beams 2 \
> logs/evaluate_vicuna_20230818.log 2>&1 &