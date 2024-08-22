#!/bin/bash
# Loading the required module
module load anaconda/2023a
python /home/gridsan/ywang5/projects/llm-context/long_context_general.py --model_path "/home/gridsan/ywang5/hf/models/vicuna-7b-v1.3" --context "news" --attack "JCB"