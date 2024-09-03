#!/bin/bash
# Loading the required module
module load anaconda/2023a

export PATH=$HOME/.local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/cuda/lib:$LD_LIBRARY_PATH

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export DISKCACHE_DIR=/home/gridsan/ywang5/hf/misc
export TMPDIR=/state/partition1/user/$USER

mkdir $TMPDIR

cls_path='/home/gridsan/ywang5/hf/models/HarmBench-Llama-2-13b-cls'
behaviors_path=$2
completions_path='/home/gridsan/ywang5/projects/llm-context/completions/completion_mistral_general_dan_harmful.json'
save_path='/home/gridsan/ywang5/projects/llm-context/results/results.json'

echo "cls_path=$cls_path"
echo "behaviors_path=$behaviors_path"
echo "completions_path=$completions_path"
echo "save_path=$save_path"

python -u evaluate_completions.py \
    --cls_path $cls_path \
    --behaviors_path $behaviors_path \
    --completions_path $completions_path \
    --save_path $save_path \
    --include_advbench_metric
