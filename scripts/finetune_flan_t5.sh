TRAIN_DATA_PATH="/mnt/data/zekai/feedback_data_hotpot_qa_train.json"
VAL_DATA_PATH="/mnt/data/zekai/feedback_data_hotpot_qa_val.json"
OUTPUT_PATH="/mnt/data/zekai/feedback_model_flan_t5"
MODEL_PATH="/mnt/data/zekai/flan-t5-large"

nohup python3 finetune_flan_t5.py \
--train_data_path $TRAIN_DATA_PATH \
--val_data_path $VAL_DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--epochs 5 \
--eval_steps 500 \
--save_steps 1500 \
--max_length 1024 \
--wandb True \
> logs/finetune_flan_t5_20231223.log 2>&1 &