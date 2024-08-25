#!/bin/bash
# Loading the required module
module load anaconda/2023a

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TRANSFORMERS_CACHE=/home/gridsan/ywang5/hf/models
export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

python /home/gridsan/ywang5/projects/llm-context/long_context_dan.py --model_path "/home/gridsan/ywang5/hf/models/mistral-7b-v0.2" --prompt_type "benign"
python /home/gridsan/ywang5/projects/llm-context/long_context_dan.py --model_path "/home/gridsan/ywang5/hf/models/mistral-7b-v0.2" --prompt_type "benign" --context "values"