#!/bin/bash

MODEL_NAME="/mnt/data/zekai/flan-t5-large"
DATA_PATH="/mnt/data/zekai/summaries_after_tune_1200.json"
SAVE_PATH="/mnt/data/zekai/qa_metric_result_1200.json"

nohup python3 metric.py \
--model_name $MODEL_NAME \
--data_path $DATA_PATH \
--save_path $SAVE_PATH \
--metrics_name "rouge" \
> logs/qa_metric_rouge_20230902_1200.log 2>&1 &