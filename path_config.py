import os

####################
CACHEDIR = os.environ['TRANSFORMERS_CACHE']  # huggingface model cache_dir
DATAPATH = "./dataset"  # tokenized data directory (containing folders e.g. metaicl, soda)
SAVEPATH = "./result"  # result directory (also have subfolders of dataset names)
####################
