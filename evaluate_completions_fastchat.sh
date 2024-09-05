#!/bin/bash
# Loading the required module
module load anaconda/2023a

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

python /home/gridsan/ywang5/projects/llm-context/evaluate_completions_fastchat.py \
    --model_path "/home/gridsan/ywang5/hf/models/mistral-7b-v0.2" \
    --completions_path "/home/gridsan/ywang5/projects/llm-context/completions/completion_mistral_generalDAN_news_0_harmful.json" \
    --save_path "/home/gridsan/ywang5/projects/llm-context/results/evaluation_mistral_generalDAN_news_0_harmful.json"\
    --include_advbench_metric