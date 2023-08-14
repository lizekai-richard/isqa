TOT_CUDA="0" #Upgrade bitsandbytes to the latest version to enable balanced loading of multiple GPUs, for example: pip install bitsandbytes==0.39.0
BASE_MODEL="lmsys/vicuna-7b-v1.3" #"decapoda-research/llama-13b-hf"
LORA_PATH="lora-Vicuna" #"./lora-Vicuna/checkpoint-final"
USE_LOCAL=1 # 1: use local model, 0: use huggingface model
TYPE_WRITER=0 # whether output streamly
if [[ USE_LOCAL -eq 1 ]]
then
cp sample/instruct/adapter_config.json $LORA_PATH
fi

#Upgrade bitsandbytes to the latest version to enable balanced loading of multiple GPUs
CUDA_VISIBLE_DEVICES=0,1
nohup python3 summarization.py \
    --model_path $BASE_MODEL \
    --lora_path $LORA_PATH \
    --use_local $USE_LOCAL \
    --use_typewriter $TYPE_WRITER \
> logs/inference_20230731.log 2>&1 &