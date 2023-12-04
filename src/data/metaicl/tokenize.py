import os
import glob
import json
import torch
from collections import defaultdict

DATAPATH = "/storage/janghyun/datasets"
DATAPATH_METAICL = "/home/janghyun/Codes/chat/metaICL"


def load_data(task, split, k, seed=100, config_split=None, datasets=None):

    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join(f"{DATAPATH_METAICL}/config", task + ".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data_dict = defaultdict(list)
    for dataset in datasets:
        data_path = os.path.join(f"{DATAPATH_METAICL}/data", dataset,
                                 "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                data_dict[dataset].append(dp)

    print(f"Load data from {task}.json with {len(datasets)} tasks")
    return data_dict


def load_tokenize_save(tokenizer, is_llama=True, chat=True):
    load_path = os.path.join(DATAPATH_METAICL, "data")
    if is_llama:
        if chat:
            save_path_base = os.path.join(DATAPATH, f"metaICL/llama2-chat")
        else:
            save_path_base = os.path.join(DATAPATH, f"metaICL/llama")
    else:
        save_path_base = os.path.join(DATAPATH, f"metaICL/flan-t5")
    print("Save data at", save_path_base)

    failed = []
    for dataset in sorted(os.listdir(load_path)):
        # if dataset not in ['kilt_hotpotqa']:
        #     continue

        files = glob.glob(f"{load_path}/{dataset}/*.jsonl")
        if len(files) > 16:
            print(f"{dataset} has more than 16 files")
            failed.append(dataset)
            continue

        save_dir = os.path.join(save_path_base, dataset)
        os.makedirs(save_dir, exist_ok=True)

        try:
            for file in files:
                if 'dev.jsonl' in file:
                    continue
                if 'test.jsonl' in file and '100_test.jsonl' not in file:
                    continue

                save_path_file = os.path.join(save_dir, os.path.basename(file))
                # if os.path.isfile(save_path_file):
                #     continue

                tokenized = {'input': [], 'output': [], 'options': [], 'length': []}
                with open(file, "r") as f:
                    for line in f:
                        dp = json.loads(line)

                        tokenized['input'].append(
                            tokenizer(dp['input'].strip(), add_special_tokens=False)['input_ids'])

                        tokenized['output'].append(
                            tokenizer(dp['output'].strip(), add_special_tokens=False)['input_ids'])

                        if len(dp['options']) == 0:
                            continue

                        tokenized['options'].append([])
                        for option in dp['options']:
                            tokenized['options'][-1].append(
                                tokenizer(option.strip(), add_special_tokens=False)['input_ids'])

                tokenized['length'] = [len(x) for x in tokenized['input']]

                assert len(tokenized['input']) == len(tokenized['output'])
                print(save_path_file, len(tokenized['input']))
                torch.save(tokenized, save_path_file)
        except KeyboardInterrupt:
            break
        except:
            failed.append(dataset)

        print()

    if len(failed) > 0:
        print(f"{failed} failed")


def test_tokenize(tokenizer, dp, k=2, is_llama=True):
    comp = '<COMP>'
    out_prefix = '\nOutput: '
    if is_llama:
        comp_token = tokenizer(comp.strip(), add_special_tokens=False)['input_ids']
    else:
        comp_token = tokenizer(comp.rstrip(), add_special_tokens=False)['input_ids']

    prefix_token = tokenizer(out_prefix.strip(), add_special_tokens=False)['input_ids']
    print("Comp token: ", comp_token)
    print("Output prefix: ", prefix_token)

    # full text tokenize
    demonstrations = []
    for i in range(k):
        demonstrations.append(dp[i]['input'].strip() + f'{out_prefix}{dp[i]["output"].strip()}')

    demon_cat = comp.join(demonstrations) + comp
    input = dp[k]['input'].strip()
    output = f'{out_prefix}{dp[k]["output"].strip()}'
    input_full = demon_cat + input + output
    input_full_token = tokenizer(input_full, add_special_tokens=False)['input_ids']

    # each text tokenize
    inputs = [
        tokenizer(dp[i]['input'].strip(), add_special_tokens=False)['input_ids']
        for i in range(k + 1)
    ]
    outputs = [
        tokenizer(dp[i]['output'].strip(), add_special_tokens=False)['input_ids']
        for i in range(k + 1)
    ]
    input_sep_token = []
    for i in range(k):
        input_sep_token += inputs[i] + prefix_token + outputs[i] + comp_token
    input_sep_token += inputs[k] + prefix_token + outputs[k]
    print('\n', tokenizer.decode(input_sep_token))

    if input_full_token != input_sep_token:
        print('\n', tokenizer.decode(input_full_token))
        print(input_full_token)
        print()
        print(input_sep_token)
        for i in range(min(len(input_full_token), len(input_sep_token))):
            if input_full_token[i] != input_sep_token[i]:
                print(i, input_full_token[i], input_sep_token[i])
                break
        raise AssertionError("Tokenization is not equal!")

    print("Test passed!")


if __name__ == '__main__':
    from transformers import AutoTokenizer, LlamaTokenizer

    is_llama = True
    chat = True
    if is_llama:
        if chat:
            tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        else:
            tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')

    tokenizer.add_special_tokens({"additional_special_tokens": ["<COMP>"]})
    tokenizer.model_max_length = 10000
    print("hi")
    print(tokenizer)

    load_tokenize_save(tokenizer, is_llama, chat)

    # task = 'hr_to_lr'

    # train_data = load_data(task, "train", k=16384)
    # test_train_data = load_data(task, "train", k=16, config_split="test")
    # test_data = load_data(task, "test", k=16, config_split="test")

    # for i, (k, v) in enumerate(train_data.items()):
    #     print(f"[Train] {k[:20]:20s}\t{len(v)}")
    # for i, (k, v) in enumerate(test_data.items()):
    #     print(f"[Dev] {k[:20]:20s}\t{len(v)}")
    # print()

    # for name, dp in train_data.items():
    #     test_tokenize(tokenizer, dp, k=2, is_llama=is_llama)
