#!/bin/bash
# Loading the required module
module load anaconda/2023a

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

export CHECKPOINT_DIR=/home/gridsan/ywang5/hf/models/Meta-Llama3.1-8B
export PYTHONPATH=/home/gridsan/ywang5/projects/llm-context:$PYTHONPATH
torchrun /home/gridsan/ywang5/projects/llm-context/evaluate_completions_llama.py

# Set path for Llama model
CHECKPOINT_DIR=/home/gridsan/ywang5/hf/models/Meta-Llama3.1-8B
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun evaluate_completions_llama.py $CHECKPOINT_DIR
export PYTHONPATH=/home/gridsan/ywang5/projects/llm-context:$PYTHONPATH
