#!/bin/bash

nohup python3 zero_shot_summ.py \
--max-new-tokens 200 \
--test_size 2000 \
--num-gpus 2 \
> logs/run_20230804.log 2>&1 &