#!/bin/bash
# Loading the required module
module load anaconda/2023a

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

# Set path for Llama model
CHECKPOINT_DIR=/home/gridsan/ywang5/hf/models/Meta-Llama3.1-8B
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun evaluate_completions_llama.py $CHECKPOINT_DIR