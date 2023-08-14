#!/bin/bash

nohup python3 zero_shot_mrc.py \
--max-new-tokens 50 \
--test_size 500 \
> logs/run_20230719.log 2>&1 &