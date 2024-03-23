# Compressed Context Memory
![main](image/main.png)

[**Paper**](https://janghyun1230.github.io/paper/ccm23.pdf) | [**arXiv**](https://arxiv.org/abs/2312.03414) | [**Project Page**](https://janghyun1230.github.io/memory/)

✨ **Main features** of our method:
- Dynamic updates of **compressed key/value memory** during LLM interactions. 
- Only requiring a **conditional LoRA for compression**.
- A **fully parallelized training** strategy for recurrent compression procedures. 
- Evaluations on diverse applications: conversation, multi-task ICL, and personalization. 
- [Update 24.02.06] Streaming setting evaluation is available.  

## Setup
```
conda create --name ccm python=3.9
conda activate ccm
pip install -r requirements.txt
```

Supported Models: **LLaMA / LLaMA-2-chat / Mistral**
> [!IMPORTANT]  
> - In [`./path_config.py`](https://github.com/snu-mllab/Context-Memory/blob/main/path_config.py), please set directory configurations.
> - To use LLaMA, please convert the LLaMA weights into Hugging Face Transformers format using the [guideline](https://huggingface.co/docs/transformers/main/model_doc/llama).
> - [Update 24.02.21] We support Mistral models! To use the model, please upgrade `pip install transformers==4.37.2 accelerate==0.27.2`
> - You can train and test models by using `--model [llama-7b,llama-2-7b-chat, mistral-7b-inst]` flags.

We release datasets and models via gdown (see below). 
> [!TIP]
> - When gdown incurs errors, please directly download files from [dataset link](https://drive.google.com/drive/folders/16bG_zCiEL27h5vVL_QN0OW3PH1iQdlf_?usp=drive_link) and [model link](https://drive.google.com/drive/folders/1qutEXBekpUTaE8fJhjKT-5DMzXpN55cx?usp=drive_link) (put model subfolders in SAVEPATH and dataset subfolders in DATAPATH from path_config.py). 

## Demo: Interactive inference with compressed memory
```
python download.py --type model --name [unified,pretrain]  # Download adapters
python inference.py -i -m [llama-7b,llama-2-7b-chat] --eval_name concat_recur
```
- **An example of an interactive chat program:**
  <div style="margin-top: 0px;"></div>
  <img src="https://github.com/snu-mllab/Context-Memory/blob/main/image/demo.png" align="center" width=55%>

- **Testing pre-defined examples**: Run the code without `-i` flag to measure perplexity on the target output and compare generation results. You can modify test examples in [`./src/test_case.py`](https://github.com/snu-mllab/Context-Memory/blob/main/src/test_case.py). 
  <div style="margin-top: 0px;"></div>
  <img src="https://github.com/snu-mllab/Context-Memory/blob/main/image/test.png" align="center" width=90%>
> [!Note]  
> - In default, download adapters with `--name unified` for llama-7b and `--name pretrain` for llama-2-7b-chat.
> - To test CCM-merge, set `--eval_name merge_recur`.
> - [Update 24.01.12] We release a compression adapter for the general purpose which is trained on the mixture of datasets including samples from [RedPajama-v2](https://www.together.ai/blog/redpajama-data-v2) and [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (# training samples is 500k). To test the adapter, download `--name pretrain` and set `--dataset pretrain` for inference.py.

## Streaming setting
- To conduct the evaluation of CCM-concat (LLaMA) in a streaming setting with a sliding window, run
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
python download.py --type data --name [metaicl,soda]
```
> [!Note]  
> - In our codes, `--dataset unified` refers to the mixture of MetaICL and SODA. Download both datasets to use this argument. 
> - To use other datasets, you should make a collator function. Check for [`./src/data`](https://github.com/snu-mllab/Context-Memory/tree/main/src/data).

## Training
> [!Important]  
> - Our experiments basically run on a single A100 80GB within 5~24h. In the case of DailyDialog, which has a smaller context length, we can run on a single RTX 3090 GPU with 24GB memory. 
> - Set up a [Wandb](https://wandb.ai/) account for logging, and replace the username with yours in the wandb.entity field of [`src/conf/config.yaml`](https://github.com/snu-mllab/Context-Memory/blob/main/src/config/config.yaml).

**Step 1 (optional): Fintuning LLaMA.** We recommend first finetuning the LLaMA pretrained models on a dataset: 
```
python run.py --train --dataset [unified,metaicl,dialog,lamp] --model llama-7b \
    --comp_type no
```
- The LoRA adapters will be saved at `{SAVEPATH}/{dataset}/llama-7b-no`. Set SAVEPATH in path_config.py.
- For aligned models such as **LLaMA-2-chat/Mistral-instruct**, it's okay to skip this step. 

**Step 2: Training a compression adapter.** 
```
python run.py --train --dataset [unified,metaicl,dialog,lamp] --model llama-7b \
    --load_path llama-7b-no \ 
    --attn_type [concat_recur,merge_recur] --n_tok [# <COMP> tokens]
```
- Default configurations for each dataset can be found in [`./src/config`](https://github.com/snu-mllab/Context-Memory/tree/05d0b542b7d6cc7339c9b13e66d4c15c600efe34/src/config). The arguments provided by the command line will overwrite the default configurations. 
- If you have skipped step 1, then execute run.py without the `--load_path` flag. 

## Evaluation
- We release optimized adapters via [Google Drive](https://drive.google.com/drive/folders/1qutEXBekpUTaE8fJhjKT-5DMzXpN55cx?usp=drive_link). To download, run
```
python download.py --type model --name [unified,pretrain,metaicl,dialog,lamp]
```
- To test models, run
```
python run.py --dataset [metaicl,dialog,lamp] --model llama-7b \
    --load_path llama-7b-no \ 
    --eval_path [path for compression adapter] \ 
    --attn_type [concat_recur,merge_recur]
```
> [!Note]  
> - Set `--train_dataset` for cross-dataset evaluation.
>   - Ex) To evaluate a model trained with a unified trainset on DailyDialog testset, set `--train_dataset unified --dataset dialog`. 
> - The parent directory of load/eval paths is `{SAVEPATH}/{args.train_dataset}`.  
>   - Ex) `--eval_path finetune/llama-7b-no-online-concat_recur-ntok2 --attn_type concat_recur` will evaluate CCM-concat (LLaMA-7B) with two compression tokens.
>   - Be aware to set the correct `--model` and `--attn_type` of the adapter. 
>   - `--n_tok` will be automatically parsed from the eval_path.
> - For LLaMA-2-chat or Mistral-instruct, we don't need the `--load_path` flag. 
> - In the case of MetaICL and LaMP, we use --attn_type [concat,merge] (see [L218-223 in run.py](https://github.com/snu-mllab/Context-Memory/blob/05d0b542b7d6cc7339c9b13e66d4c15c600efe34/run.py#L218C3-L218C3)). To aggregate evaluation results on multiple test tasks, run `parse_results_metaicl.py --dataset [unified,metaicl] --folder ['',finetune]`.

## Reference
- This code is created based on the [Gisting repository](https://github.com/jayelm/gisting).

## Citation
```
@inproceedings{
      kim2024compressed,
      title={Compressed Context Memory for Online Language Model Interaction},
      author={Jang-Hyun Kim and Junyoung Yeom and Sangdoo Yun and Hyun Oh Song},
      booktitle={ICLR},
      year={2024},
}
```
