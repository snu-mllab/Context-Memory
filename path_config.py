import os

####################
CACHEDIR = "/storage/shared/janghyun"  # os.environ['TRANSFORMERS_CACHE']  # huggingface model cache_dir (e.g., LLaMA-2)
LLAMADIR = "/storage/shared/janghyun"  # LLaMA model directory (llama-7b-hf)
DATAPATH = "./dataset"  # tokenized data directory (containing folders e.g. metaicl, soda)
SAVEPATH = "./result"  # result directory (containing folders of dataset names)
####################

# DATAPATH example
## DATAPATH
## |- metaicl
## |- soda

# SAVEPATH example
## SAVEPATH
## |- all
##  |- llama-7b-no
##  |- finetune
## |- metaicl
## |- dialog


def model_path(model_name):
    if model_name == "llama-7b":
        path = os.path.join(LLAMADIR, "llama-7b-hf")

    elif model_name == "llama-13b":
        path = os.path.join(LLAMADIR, "llama-13b-hf")

    elif model_name == "llama-2-7b-chat":
        path = "meta-llama/Llama-2-7b-chat-hf"

    elif model_name == "llama-2-13b-chat":
        path = "meta-llama/Llama-2-13b-chat-hf"

    elif model_name == "llama-2-7b":
        path = "meta-llama/Llama-2-7b-hf"

    elif model_name == "llama-2-13b":
        path = "meta-llama/Llama-2-13b-hf"

    elif model_name == "mistral-7b":
        path = "mistralai/Mistral-7B-v0.1"

    elif model_name == "mistral-7b-inst":
        path = "mistralai/Mistral-7B-Instruct-v0.2"

    elif "flan-t5" in model_name:
        path = f"google/{model_name}"

    elif model_name == "llama-debug":
        path = "meta-llama/Llama-2-7b-chat-hf"

    elif model_name == "mistral-debug":
        path = "mistralai/Mistral-7B-Instruct-v0.2"

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return path


def map_config(model_name):
    is_llama = "llama" in model_name.lower() or "mistral" in model_name.lower()

    if "debug" in model_name:
        config = "llama-debug"
    elif is_llama and "7b" in model_name:
        config = "llama-7b"
    elif is_llama and "13b" in model_name:
        config = "llama-13b"
    elif "flan-t5" in model_name:
        config = model_name
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return config
