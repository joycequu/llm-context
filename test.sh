export HF_HOME=/state/partition1/user/$(whoami)/hf
mkdir -p "$HF_HOME"

python /home/gridsan/ywang5/projects/llm-context/long_context.py