#!/bin/bash

nohup python3 zero_shot_mrc.py \
--max-new-tokens 50 \
--test_size 2620 \
--dataset "../results/data/scimrc_hard.json" \
> logs/run_scimrc_hard_20230720.log 2>&1 &