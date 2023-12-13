DATA_PATH="/mnt/data/zekai/generator_data.json"
MODEL_PATH="/mnt/data/zekai/vicuna_7b"
OUTPUT_PATH="correction_results_300_new.json"

nohup python3 baseline.py \
--model_path $MODEL_PATH \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--max_length 2048 \
--max-new-tokens 200 \
--min_new_tokens 100 \
--batch_size 4 \
> logs/run_20230804.log 2>&1 &