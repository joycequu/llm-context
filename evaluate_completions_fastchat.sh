#!/bin/bash
# Loading the required module
module load anaconda/2023a

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

python /home/gridsan/ywang5/projects/llm-context/evaluate_completions_fastchat.py \
    --model_path "/home/gridsan/ywang5/hf/models/Meta-Llama-3.1-8B" \
    --completions_path "/home/gridsan/ywang5/projects/llm-context/harmful_testing/completions/completion_mistral_generalDAN_news_0_harmful.json" \
    --save_path "/home/gridsan/ywang5/projects/llm-context/harmful_testing/results/evaluation_llama3.1_generalDAN_news_0_harmful.json"\
    --include_advbench_metric