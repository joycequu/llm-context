#!/bin/bash
# Loading the required module
conda activate env8
module load anaconda/2023a

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TRANSFORMERS_CACHE=/home/gridsan/ywang5/hf/models
export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

python /home/gridsan/ywang5/projects/llm-context/long_context_dan.py --model_path "/home/gridsan/ywang5/hf/models/vicuna-7b-v1.3"