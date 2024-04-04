DATA_PATH="/mnt/data/zekai/generator_data.json"
BASE_MODEL_PATH="/mnt/data/zekai/llama2-7b-chat-hf"
LORA_PATH="/mnt/data/zekai/feedback_model_7b/checkpoint-600"
FEEDBACK_MODEL_PATH="/mnt/data/zekai/vicuna-7b-v1.3"
OUTPUT_PATH="iter_refine_results_llama2_300_592.json"

nohup python3 iter_refine_on_feedback.py \
--data_path $DATA_PATH \
--base_model_path $BASE_MODEL_PATH \
--feedback_model_path $FEEDBACK_MODEL_PATH \
--lora_path $LORA_PATH \
--save_path $OUTPUT_PATH \
--max_length 2048 \
--feedback_max_length 512 \
--threshold 0.5 \
--num_beams 2 \
--num_correction_steps 6 \
--correction_batch_size 4 \
> logs/iter_refine_results_llama2_300_592.log 2>&1 &
