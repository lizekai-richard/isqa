TOT_CUDA="0,1,2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="../data/scimrc_for_inst_tune_sec_based.json"
OUTPUT_PATH="Vicuna-LoRA"
MODEL_PATH="../llm_weights/vicuna-7b-v1.3"
TEST_SIZE=800

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE
# > logs/peft_vicuna_20230731.log 2>&1 &
