export HDF5_USE_FILE_LOCKING='FALSE'
export HF_HOME=/state/partition1/user/$(whoami)/hf
mkdir -p $(HF_HOME)

python download_model.py