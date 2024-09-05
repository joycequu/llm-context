#!/bin/bash
# Loading the required module
module load anaconda/2023a

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

behaviors_path=$2
completions_path=$3
save_path=$4

echo "behaviors_path=$behaviors_path"
echo "completions_path=$completions_path"
echo "save_path=$save_path"


python evaluate_completions_fastchat.py \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --save_path $save_path \
    --include_advbench_metric

python /home/gridsan/ywang5/projects/llm-context/evaluate_completions_fastchat.py \
    --model_path "/home/gridsan/ywang5/hf/models/mistral-7b-v0.2" \
    --completions_path "/home/gridsan/ywang5/projects/llm-context/completions/completion_mistral_generalDAN_news_0_harmful.json" \
    --save_path "/home/gridsan/ywang5/projects/llm-context/results/evaluation_mistral_generalDAN_news_0_harmful.json"\
    --include_advbench_metric