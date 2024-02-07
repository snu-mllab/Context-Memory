# Compressed Context Memory
![main](image/main.png)

[**Paper**](https://janghyun1230.github.io/paper/ccm23.pdf) | [**arXiv**](https://arxiv.org/abs/2312.03414) | [**Project Page**](https://janghyun1230.github.io/memory/)

- Our approach dynamically creates **compressed key/value memory** during LLM interactions. 
- Our approach only requires a **conditional LoRA for compression**.
- We propose a **fully parallelized training** strategy for recurrent compression procedures. 
- We support evaluations on diverse applications: conversation, multi-task ICL, and personalization. 
- [Update 24.02.06] Streaming setting codes updated.  

## Setup
```
conda create --name ccm python=3.9
conda activate ccm
pip install -r requirements.txt
```
- We use PyTorch 2.0.0.

Supported Models: **LLaMA / LLaMA-2-chat**
- Please convert the LLaMA weights into Hugging Face Transformers format using the [guideline](https://huggingface.co/docs/transformers/main/model_doc/llama).
- In [`./path_config.py`](https://github.com/snu-mllab/Context-Memory/blob/main/path_config.py), please set directory configurations.

We provide download codes for datasets and models (see below). 
- When gdown incurs errors, please directly download files from [dataset link](https://drive.google.com/drive/folders/16bG_zCiEL27h5vVL_QN0OW3PH1iQdlf_?usp=drive_link) and [model link](https://drive.google.com/drive/folders/1qutEXBekpUTaE8fJhjKT-5DMzXpN55cx?usp=drive_link) (put model subfolders in SAVEPATH and dataset subfolders in DATAPATH from path_config.py). 

## Demo: Interactive inference with compressed memory
```
python download.py --type model --name unified  # Download adapters
python inference.py -i -m llama-7b --eval_name [concat_recur/merge_recur]
```
- This will launch an interactive chat system based on LLaMA-7B:  
  <img src="https://github.com/snu-mllab/Context-Memory/blob/main/image/demo.png" align="center" width=55%>
- To test with pre-defined examples, run the code without `-i` flag. You can modify test examples in [`./src/test.py`](https://github.com/snu-mllab/Context-Memory/blob/main/src/test.py).
  <img src="https://github.com/snu-mllab/Context-Memory/blob/main/image/test.png" align="center" width=90%>
- [Update 24.01.12] We release a compression adapter for the general purpose which is trained on the mixture of datasets including samples from [RedPajama-v2](https://www.together.ai/blog/redpajama-data-v2) and [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (# training samples is 500k). To test the adapter, download `--name pretrain` and set `--dataset pretrain` for inference.py.

## Streaming setting
- To conduct evaluation of CCM in a streaming setting with sliding window, run
```
python download.py --type data --name pg19
python download.py --type model --name pretrain
python inference.py --stream 
```
<img src="https://github.com/snu-mllab/Context-Memory/blob/main/image/stream.png" align="center" width=90%>


## Dataset 
- We provide tokenized data of [MetaICL](https://github.com/facebookresearch/MetaICL) and [SODA](https://github.com/skywalker023/sodaverse) for LLaMA. Smaller datasets, e.g., DailyDialog, will be downloaded and tokenized automatically during training. 
- To download tokenized datasets, run
```
python download.py --type data --name [metaicl/soda]
```
- In our codes, `--dataset unified` refers to the mixture of MetaICL and SODA. Download both datasets to use this argument. 
- To use other datasets, you should make a collator function. Check for `./src/data`.

## Training
- Our experiments basically run on a single A100 80GB within 5~24h. In the case of DailyDialog, which has a smaller context length, we can run on a single RTX 3090 GPU. 
- Set up a [Wandb](https://wandb.ai/) account for logging, and replace the username with yours in the wandb.entity field of `src/conf/config.yaml`.
- We recommend first finetuning the LLaMA pretrained models on a dataset: 
```
python run.py --train --dataset [unified/metaicl/dialog/lamp] --model llama-7b \
    --comp_type no
```
- The LoRA adapters will be saved at `{SAVEPATH}/{dataset}/llama-7b-no`. Set SAVEPATH in path_config.py.
- Then we train our compression adapter as
```
python run.py --train --dataset [unified/metaicl/dialog/lamp] --model llama-7b \
    --load_path llama-7b-no \ 
    --attn_type [concat_recur/merge_recur] --n_tok [# <COMP> tokens]
```
- Default configurations for each dataset can be found in [`./src/config`](https://github.com/snu-mllab/Context-Memory/tree/05d0b542b7d6cc7339c9b13e66d4c15c600efe34/src/config). The arguments provided by the command line will overwrite the default configurations. 
- For aligned models such as LLaMA-2-chat, it's okay to skip the previous finetuning step with `--comp_type no`. In this case, run the training codes without `--load_path`. 

## Evaluation
- We release optimized adapters via [Google Drive](https://drive.google.com/drive/folders/1qutEXBekpUTaE8fJhjKT-5DMzXpN55cx?usp=drive_link). To download, run
```
python download.py --type model --name [unified/metaicl/dialog/lamp/pretrain]
```
- To test models, run
```
python run.py --dataset [metaicl/dialog/lamp] --model llama-7b \
    --load_path llama-7b-no \ 
    --eval_path [path for compression adapter] \ 
    --attn_type [concat_recur/merge_recur]
```
- Set `--train_dataset` for cross-dataset evaluation, e.g., to evaluate a model trained with an unified trainset on DailyDialog testset, set `--train_dataset unified --dataset dialog`. 
- The parent directory of --load_path and --eval_path is `{SAVEPATH}/{args.dataset}`.
  - The argument `--n_tok` will be automatically parsed from the path name.
  - As an example, `--eval_path finetune/llama-7b-no-online-concat_recur-ntok2 --attn_type concat_recur` will test CCM-concat with two compression tokens trained with --dataset.
  - Be aware to set the correct `--attn_type` of the adapter. 
- In the case of MetaICL/LaMP, we use --attn_type [concat/merge] (see [L218-223 in run.py](https://github.com/snu-mllab/Context-Memory/blob/05d0b542b7d6cc7339c9b13e66d4c15c600efe34/run.py#L218C3-L218C3)). To aggregate evaluation results on multiple test tasks, run `parse_results_metaicl.py --dataset [unified,metaicl] --folder ['',finetune]`.

## Reference
- This code is created based on the [Gisting repository](https://github.com/jayelm/gisting).

## Citation
```
@article{kim2023compressed,
      title={Compressed Context Memory For Online Language Model Interaction}, 
      author={Kim, Jang-Hyun and Yeom, Junyoung and Yun, Sangdoo and Song, Hyun Oh},
      journal={arXiv preprint arXiv:2312.03414},
      year={2023},
}
```
