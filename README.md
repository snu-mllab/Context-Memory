# Compressed Context Memory
![main](image/main.png)

[**Paper**](https://janghyun1230.github.io/) | [**Project Page**](https://janghyun1230.github.io/)

- Our approach dynamically creates compressed memory of contexts during LLM interactions. 
- Our approach only requires to train a **conditional LoRA for compression**. 
- We use a **fully parallelized training** strategy for recurrent compression procedures. 
- We conduct evaluation on diverse applications: conversation, multi-task ICL, and personalization. 

## Setup
```
conda create --name ccm python=3.9
conda activate ccm
pip install -r requirements.txt
```
- We use PyTorch 2.0.0.

Supported Models: LLaMA-7B / LLaMA-2-7B-chat
- Please convert the LLaMA weights into Hugging Face Transformers format using the [guideline](https://huggingface.co/docs/transformers/main/model_doc/llama).
- In [./path_config.py](https://github.com/snu-mllab/Context-Memory/blob/main/path_config.py), please set directory configurations.


## Demo: Interactive inference with compressed memory
```
conda create --name ccm python=3.9
conda activate ccm
pip install -r requirements.txt
```


## Dataset 
- 
- 

## Training

- Training full-attention model
```
python run.py --train \
    --model llama-7b \
    --k 16 --comp_type no \
```
- Training CCM-{concat,merge} model
```
python run.py --train \
    --model llama-7b \
    --k 16 \
    --attn_type {concat,merge} --n_tok 8 \
    --load_path llama-7b-no-k16
```
- Evaluating full-attention model
```
python run.py \
    --model llama-7b \
    --k {1,2,4,8,16} --comp_type no \
    --eval_path llama-7b-no-k16
```
- Evaluating CCM-{concat,merge} model
```
python run.py \
    --model llama-7b \
    --k {1,2,4,8,16} --comp_type no \
    --load_path llama-7b-no-k16 \
    --eval_path finetune/llama-7b-no-k16-cond_lora-sepembed-norel-recur-{concat,merge}-g8-k16
```

## Evaluation


## Citation
