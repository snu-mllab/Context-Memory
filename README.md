# Compressed Context Memory
![main](image/main.png)

[**Paper**](https://janghyun1230.github.io/) | [**Project Page**](https://janghyun1230.github.io/)

- Our approach dynamically creates **compressed memory of contexts** during LLM interactions. 
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

Supported Models: LLaMA / LLaMA-2-chat
- Please convert the LLaMA weights into Hugging Face Transformers format using the [guideline](https://huggingface.co/docs/transformers/main/model_doc/llama).
- In [`./path_config.py`](https://github.com/snu-mllab/Context-Memory/blob/main/path_config.py), please set directory configurations.


## Demo: Interactive inference with compressed memory
```
python download.py --type model --dataset all  # Download adapters
python interact.py -i -m llama-7b --eval_name [concat_recur/merge_recur]
```
- This will launch an interactive chat system based on LLaMA-7B:  
<img src="https://github.com/snu-mllab/Context-Memory/tree/main/image/demo.png" align="center" width=50%>

## Dataset 
- We provide tokenized data of [MetaICL](https://github.com/facebookresearch/MetaICL) and [SODA](https://github.com/skywalker023/sodaverse) for LLaMA. Smaller datasets including DailyDialog will be downloaded and tokenized automatically. 
- To download tokenized datasets, run
```
python download.py --type data --dataset [metaicl/soda]
```
- To use other datasets, you should make a collator function. Check for `./src/data`.

## Training
- Our experiments basically run on a single A100 GPU. In the case of DailyDialog which has a smaller context length, we can run on a single RTX 3090 GPU. 
- Set up a [Wandb](https://wandb.ai/) account for logging, and replace username with yours in the wandb.entity field of `src/conf/config.yaml`.
- We recommand to first finetune LLaMA pretrained models on a dataset: 
```
python run.py --train --dataset [all/metaicl/dialog] --model llama-7b \
    --comp_type no
```
- The 'all' dataset refers to the mixture of MetaICL and SODA.
- The LoRA adapters will be saved at `{SAVEPATH}/{dataset}/llama-7b-no`. Set SAVEPATH in path_config.py.
- Then we train our compression adapter as
```
python run.py --train --dataset [all/metaicl/dialog] --model llama-7b \
    --load_path llama-7b-no \ 
    --attn_type [concat_recur/merge_recur] --n_tok [# <COMP> tokens]
```
- Default configurations for each dataset can be found in `./src/config`. The arguments provided by command line will overwrite the default configurations. 
- For aligned model such as LLaMA-2-chat, it's okay to skip the previous finetuning step with `--comp_type no`. In this case, run the training codes without `--load_path`. 

## Evaluation
- We release optimized adapters via Google Drive. To download, run
```
python download.py --type model --dataset [all/metaicl/soda]
```
- To test models, run
```
python run.py --train --dataset [all/metaicl/dialog] --model llama-7b \
    --load_path llama-7b-no \ 
    --eval_path [path for compression adapter] \ 
    --attn_type [concat_recur/merge_recur]
```
- The base directory of --load_path and --eval_path is `{SAVEPATH}/{dataset}`. 
- For example, `--eval_path finetune/llama-7b-no-online-concat_recur-ntok2 --attn_type concat_recur` will test CCM-concat with two compression tokens. `--n_tok` argument is automatically parsed. Be aware to set correct `--attn_type` for the adapter. 
- In the case of MetaICL, run `parse_results_metaicl.py --dataset [all,metaicl] --folder ['',finetune]` to aggregate evaluation results on multiple test tasks.

## Reference
- This code is created based on the [Gisting repository](https://github.com/jayelm/gisting).

## Citation
