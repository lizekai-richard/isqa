DATA_PATH="/mnt/data/zekai/generator_data.json"
LORA_PATH="/mnt/data/zekai/feedback_model/checkpoint-600"
MODEL_PATH="/mnt/data/zekai/vicuna_7b"
OUTPUT_PATH="correction_results_300_new.json"

nohup python3 nl_correction.py \
--data_path $DATA_PATH \
--model_path $MODEL_PATH \
--lora_path $LORA_PATH \
--save_path $OUTPUT_PATH \
--max_length 2048 \
--feedback_max_length 512 \
--num_beams 2 \
> logs/self_correction_20231004.log 2>&1 &
