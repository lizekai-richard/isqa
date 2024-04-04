OPENAI_API_KEY="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"
DATA_PATH="/mnt/data/zekai/generator_data.json"
BASE_MODEL_PATH="/mnt/data/zekai/llama2-7b-chat-hf"
OUTPUT_PATH="iter_refine_gpt_results_llama2_5.json"

nohup python3 iter_refine_gpt.py \
--data_path $DATA_PATH \
--base_model_path $BASE_MODEL_PATH \
--save_path $OUTPUT_PATH \
--openai_api_key $OPENAI_API_KEY \
--max_length 2048 \
--feedback_max_length 512 \
--threshold 0.5 \
--num_beams 2 \
--num_correction_steps 4 \
--correction_batch_size 4 \
> logs/iter_refine_gpt_results_llama2_20230112.log 2>&1 &