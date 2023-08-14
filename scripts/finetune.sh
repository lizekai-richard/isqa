TOT_CUDA="2,3"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="sample/instruct/scimrc_for_inst_tune.json"
OUTPUT_PATH="lora-Vicuna"
# MODEL_PATH="lmsys/vicuna-7b-v1.3"
MODEL_PATH="lmsys/vicuna-7b-v1.3"
lora_checkpoint="lora-Vicuna/checkpoints"
TEST_SIZE=800

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 100 \
--save_steps 100 \
--test_size $TEST_SIZE
# > logs/peft_vicuna_20230731.log 2>&1 &
