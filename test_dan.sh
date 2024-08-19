#!/bin/bash
# Loading the required module
module load anaconda/2023a
# python /home/gridsan/ywang5/projects/llm-context/long_context_dan.py --model_path "/home/gridsan/ywang5/hf/models/vicuna-7b-v1.3" --context "values" --attack "JCB"

# DAN, no long-context is equivalent to when context_length = 0
# python /home/gridsan/ywang5/projects/llm-context/long_context_dan.py --model_path "/home/gridsan/ywang5/hf/models/vicuna-7b-v1.3" --prompt_type "benign"
# python /home/gridsan/ywang5/projects/llm-context/long_context_dan.py --model_path "/home/gridsan/ywang5/hf/models/vicuna-7b-v1.3" --prompt_type "harmful" --context "values"
# python /home/gridsan/ywang5/projects/llm-context/long_context_dan.py --model_path "/home/gridsan/ywang5/hf/models/vicuna-7b-v1.3" --prompt_type "benign" --context "values"

python /home/gridsan/ywang5/projects/llm-context/long_context_dan.py --model_path "/home/gridsan/ywang5/hf/models/vicuna-7b-v1.3"