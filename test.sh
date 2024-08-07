export HF_HOME=/state/partition1/user/$(whoami)/hf
mkdir -p $(HF_HOME)

python long_context.py