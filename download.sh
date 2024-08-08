export HDF5_USE_FILE_LOCKING='FALSE'
export HF_HOME=/state/partition1/user/$(whoami)/hf
mkdir -p $(HF_HOME)
export HF_DATASETS_CACHE=/home/gridsan/ywang5/hf/datasets
export TRANSFORMERS_CACHE=/home/gridsan/ywang5/hf/models
export HUGGINGFACE_TOKEN='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'

python download_model.py