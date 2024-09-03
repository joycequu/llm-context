#!/bin/bash
# Loading the required module
module load anaconda/2023a

export HF_HOME=/home/gridsan/ywang5/hf/misc
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR

echo "Starting news context..."
python /home/gridsan/ywang5/projects/llm-context/generate_completions_lc_dan.py --model_path "/home/gridsan/ywang5/hf/models/mistral-7b-v0.2"

echo "Starting values context..."
python /home/gridsan/ywang5/projects/llm-context/generate_completions_lc_dan.py --model_path "/home/gridsan/ywang5/hf/models/mistral-7b-v0.2" --context "values"