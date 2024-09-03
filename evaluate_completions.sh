#!/bin/bash
# Loading the required module
module load anaconda/2023a

export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

export NEW_DISK_CACHE_DIR=/state/partition1/user/$USER/diskcache
mkdir -p $NEW_DISK_CACHE_DIR
export DISKCACHE_DIR=$NEW_DISK_CACHE_DIR

export CACHE_DIR=$TMPDIR
export HF_HOME=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR

export PATH=$HOME/.local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/cuda/lib:$LD_LIBRARY_PATH

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
