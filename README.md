Pytorch implementation for "Compressed Context Memory For Online Language Model Interaction"

## Training and evaluation on MetaICL dataset ([download link](https://github.com/facebookresearch/MetaICL))

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

## Training and evaluation on LaMP dataset ([download link](https://lamp-benchmark.github.io/download))


- Training full-attention model
```
python run.py --train \
    --model llama-7b \
    --data lamp2 \
    --n_tok 16 \
    --comp_type no
```

- Training CCM-{concat,merge} model
```
python run.py --train \
    --model llama-7b \
    --data lamp2 \
    --n_tok 16 \
    --attn_type {concat,merge}
```

- Evaluating full attention model
```
python run.py \
    --model llama-7b \
    --data lamp2 \
    --n_tok {1,2,4,8,16} \
    --comp_type no \
    --eval_path auto
```

- Evaluating CCM-{concat,merge} model
```
python run.py \
    --model llama-7b \
    --data lamp2 \
    --n_tok {1,2,4,8,16} \
    --attn_type {concat,merge} \
    --load_path auto \
    --eval_path auto
```

## Training and evaluation on DailyDialog dataset ([download link](https://huggingface.co/datasets/daily_dialog))

- Training full-attention model
```
python run.py --train \
    --model llama-7b \
    --dataset dialog \
```

- Traning CCM-{concat,merge} model
```
python run.py --train \
    --model llama-7b \
    --dataset dialog \
    --attn_type {concat,merge} \
    --n_tok 8
```

- Evaluating full-attention model
```
python run.py \
    --model llama-7b \
    --dataset dialog \
    --eval_path llama-7b-no-k16
```

- Evaluating CCM-{concat,merge} model
```
python run.py \
    --model llama-7b \
    --dataset dialog \
    --attn_type {concat,merge} \
    --n_tok 8 \
    --load_path llama-7b-no-k16 \
    --eval_path finetune/llama-7b-no-k16-gistlora-sepembed-norel-recur-{concat,merge}-g8-k16
```
