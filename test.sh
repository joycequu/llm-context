export TMPDIR=/state/partition1/user/$USER
mkdir $TMPDIR
export TRANSFORMERS_CACHE=/home/gridsan/ywang5/hf/models
export HF_HOME=/state/partition1/user/$(whoami)/hf
mkdir -p "$HF_HOME"

python /home/gridsan/ywang5/projects/llm-context/long_context.py