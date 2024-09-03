#!/bin/bash
# Loading the required module
module load anaconda/2023a

# point cuda
export CUDA_HOME=/usr/local/pkg/cuda/cuda-11.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# point cudnn
export CUDNN_HOME=~/local_cuda
export LD_LIBRARY_PATH=$CUDNN_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDNN_HOME/include:$CPLUS_INCLUDE_PATH

export TMPDIR=/state/partition1/user/$USER/evaluate
mkdir $TMPDIR

export NEW_DISK_CACHE_DIR=/state/partition1/user/$USER/diskcache
mkdir -p $NEW_DISK_CACHE_DIR
export DISKCACHE_DIR=$NEW_DISK_CACHE_DIR

export HF_HOME=/state/partition1/user/$(whoami)/hf
mkdir -p $HF_HOME

cls_path='/home/gridsan/ywang5/hf/models/HarmBench-Llama-2-13b-cls'
behaviors_path='/home/gridsan/ywang5/projects/llm-context/data/behavior_datasets/harmbench_behaviors_text_all.csv'
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
