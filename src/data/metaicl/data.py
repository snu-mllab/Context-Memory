""" The original code is created by Jang-Hyun Kim.
    GitHub Repository: https://github.com/snu-mllab/Context-Memory
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset, DatasetDict
from ..utils import nested_select
from path_config import DATAPATH


def check_dp(dp):
    assert len(dp['input']) > 0
    assert len(dp['input']) == len(dp['output'])
    assert len(dp['input']) == len(dp['options']) or len(dp['options']) == 0


def delete_duplicates(options):
    a = []
    for opt in options:
        if opt not in a:
            a.append(opt)
    return a


def load_jsonl_with_options(path, dataset, is_train=True):
    dp = torch.load(path)
    check_dp(dp)
    if len(dp['options']) == 0:
        dp['options'] = [[] for _ in range(len(dp['input']))]

    n = len(dp['input'])
    dp['task'] = [dataset for _ in range(n)]
    dp['idx'] = [i for i in range(n)]
    dp['is_train'] = [is_train for _ in range(n)]

    return dp


def load_tokenized_data(meta_task="hr_to_lr",
                        is_llama=True,
                        model_name="llama-7b",
                        eval_train_task=False):
    """Load tokenized data from the path
    Returns: train_dict, eval_dict, eval_demon_dict
    """
    if is_llama:
        if "llama" in model_name.lower():
            path_base = os.path.join(DATAPATH, f"metaicl/llama")
        elif "mistral" in model_name.lower():
            path_base = os.path.join(DATAPATH, f"metaicl/mistral")
    else:
        path_base = os.path.join(DATAPATH, f"metaicl/flan-t5")

    path_config = "./src/data/metaicl"
    with open(os.path.join(path_config, meta_task + ".json"), "r") as f:
        config = json.load(f)
    assert 'train' in config.keys() and 'test' in config.keys()

    train_dict = {}
    for dataset in tqdm(config['train'], desc='Load train data'):
        try:
            train_path = os.path.join(f"{path_base}", dataset, f"{dataset}_16384_100_train.jsonl")
            dp = load_jsonl_with_options(train_path, dataset, is_train=True)
            train_dict[dataset] = dp
        except:
            print(f"Error in loading {train_path}")

    eval_dict = {}
    eval_demon_dict = defaultdict(dict)
    test_key = 'test' if not eval_train_task else 'train'
    for dataset in tqdm(config[test_key], desc='Load eval data'):
        try:
            eval_path = os.path.join(f"{path_base}", dataset, f"{dataset}_16_100_test.jsonl")
            dp = load_jsonl_with_options(eval_path, dataset, is_train=False)
            eval_dict[dataset] = dp

            for seed in [13, 21, 42, 87, 100]:
                eval_path = os.path.join(f"{path_base}", dataset,
                                         f"{dataset}_16_{seed}_train.jsonl")
                dp = load_jsonl_with_options(eval_path, dataset, is_train=False)
                eval_demon_dict[seed][dataset] = dp

        except:
            print(f"Error in loading {eval_path}")

    print(f"Load {meta_task} with {len(train_dict)} train tasks and {len(eval_dict)} test tasks")

    return train_dict, eval_dict, eval_demon_dict


def add_options(eval_dict, verbose=False):
    """change outputs to options
    Returns: option augmented data_dict
    """
    if verbose:
        print(f"\n* Adding options to inputs")

    data_dict_ = {}
    for task, dp in eval_dict.items():
        assert np.all([len(opt) for opt in dp['options']]), f"Some options are empty in {task}"

        n_opt_total = 0
        n = len(dp['input'])
        dp_option = defaultdict(list)
        for i in range(n):
            options = delete_duplicates(dp['options'][i])
            n_opt = len(options)
            n_opt_total += n_opt

            dp_option['input'] += [dp['input'][i] for _ in options]
            dp_option['output'] += [opt for opt in options]

            for key in ['task', 'idx', 'is_train', 'truncate_output']:
                dp_option[key] += [dp[key][i] for _ in range(n_opt)]

            answer = [opt == dp['output'][i] for opt in options]
            dp_option['is_answer'] += answer
            assert sum(answer) == 1, print(task, i, options, dp['output'][i])

        data_dict_[task] = dp_option

        if verbose:
            print(f"{task[:20]:20s}: {n_opt_total/n:.1f} options")

    return data_dict_


def quantile(arr, desc=''):
    arr = np.array(arr)
    print(f"{desc}: ({arr.min()}, {np.quantile(arr, 0.25):.0f}, "
          f"{np.quantile(arr, 0.5):.0f}, {np.quantile(arr, 0.75):.0f}, {arr.max()})")


def analyze_dataset_size(data_dict):
    print()
    for task, dp in data_dict.items():
        print(f"{task[:20]:20s}: {len(dp['input'])}")


def analyze_data_length(tokenizer, data_dict, max_length=256):
    print(f"\n* Analyze data with max length {max_length} for {len(data_dict)} tasks")

    for task, dp in data_dict.items():
        n = len(dp['input'])
        input_length = [len(dp['input'][i]) for i in range(n)]
        output_length = [len(dp['output'][i]) for i in range(n)]
        total_length = [input_length[i] + output_length[i] for i in range(n)]

        if max(total_length) > max_length - 2:
            print(f"{task}")
            satisfied = len([l for l in total_length if l <= max_length - 2])
            print(f" {satisfied} / {len(total_length)} data satisfied")

            quantile(input_length, desc=' Input')
            quantile(output_length, desc=' Output')
            # print(" Input: ", tokenizer.decode(dp['input'][0]))
            # print(" Output: ", tokenizer.decode(dp['output'][0]))
        else:
            continue

        print()


class MetaICLData:

    def __init__(
        self,
        tokenizer,
        is_llama,
        max_example_length=256,
        remove=True,
        comp_token=[],
        online=False,
        sep_special=False,
        add_comp_token=True,
        eval_train_task=False,
        meta_task="hr_to_lr",
    ):
        self.tokenizer = tokenizer
        self.meta_task = meta_task
        self.max_example_length = max_example_length
        self.online = online
        self.sep_special = sep_special
        self.model_name = tokenizer.name_or_path.split('/')[-1]

        self.train_dict, self.eval_dict, self.eval_demon_dict = load_tokenized_data(
            meta_task=meta_task,
            is_llama=is_llama,
            model_name=self.model_name,
            eval_train_task=eval_train_task)

        self.bos_token = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        self.eos_token = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        # Special tokens
        self.input_prefix = self.output_prefix = []
        self.comp_token = comp_token

        if type(self.comp_token) == int:
            self.comp_token = [self.comp_token]
        if not add_comp_token:
            self.comp_token = []
        print(f"MetaICL data with comp token: {self.comp_token}, online: {self.online}")

        if not self.sep_special:
            input_prefix_text = 'Input:'
            self.input_prefix = tokenizer.encode(input_prefix_text, add_special_tokens=False)
        else:
            self.input_prefix = self.bos_token

        if 'inst' not in self.meta_task:
            out_prefix_text = 'Output:'
            self.output_prefix = tokenizer.encode(out_prefix_text, add_special_tokens=False)

        # Preprocess data
        offset = len(self.input_prefix) + len(self.output_prefix)
        self.train_dict = self._truncate_long_inputs(self.train_dict,
                                                     max_len=max_example_length,
                                                     remove=remove,
                                                     offset=offset)
        self.train_dataset = self._merge_trainset(self.train_dict)
        self.train_dataset = Dataset.from_dict(self.train_dataset, split='train')

        self.eval_dict = self._truncate_long_inputs(self.eval_dict,
                                                    max_len=max_example_length,
                                                    remove=remove,
                                                    offset=offset)
        if not eval_train_task:
            self.eval_dataset = DatasetDict({
                k: Dataset.from_dict(v, split='eval_' + k) for k, v in self.eval_dict.items()
            })

            self.eval_dict_option = add_options(self.eval_dict, verbose=True)
            self.eval_dataset_option = DatasetDict({
                k: Dataset.from_dict(v, split='eval_' + k) for k, v in self.eval_dict_option.items()
            })

            for seed in self.eval_demon_dict.keys():
                self.eval_demon_dict[seed] = self._truncate_long_inputs(self.eval_demon_dict[seed],
                                                                        max_len=max_example_length,
                                                                        remove=remove,
                                                                        offset=offset,
                                                                        verbose=True)

        else:
            self.eval_dataset_option = self.eval_dict_option = None
            self.eval_demon_dict[100] = self._merge_eval_demon_dict(self.eval_demon_dict)
            self.eval_demon_dict[100] = self._truncate_long_inputs(self.eval_demon_dict[100],
                                                                   max_len=max_example_length,
                                                                   remove=remove,
                                                                   offset=offset,
                                                                   verbose=True)

            self.eval_dict = {
                k: v
                for k, v in self.eval_dict.items()
                if (len(v['input']) >= 50) and (len(self.eval_demon_dict[100][k]['input']) >= 4)
            }
            print(f"Select {len(self.eval_dict)} eval tasks with more than 50 examples, 4 demons")

            self.eval_dataset = DatasetDict({
                k: Dataset.from_dict(v, split='eval_' + k) for k, v in self.eval_dict.items()
            })

            max_eval_num = 1000
            self.eval_dataset = self.eval_dataset.shuffle(seed=0)
            self.eval_dataset = nested_select(self.eval_dataset, max_eval_num)
            n_eval = sum([len(v) for v in self.eval_dataset.values()])
            print(f"Select {max_eval_num} samples per task (total {n_eval})")

    def _truncate_long_inputs(self, data_dict, max_len=256, remove=True, verbose=True, offset=0):
        """Truncate long inputs of data_dict
        Returns: truncated data_dict
        """
        if verbose:
            print(f"\n* Truncate long inputs with max example length {max_len}")
        total_all = 0
        total_nc = 0
        for task, dp in data_dict.items():
            dp_ = defaultdict(list)
            nc = 0
            idx_cur = 0
            total = len(dp['input'])
            for i in range(total):
                input_token, output_token = dp['input'][i], dp['output'][i]
                options = dp['options'][i]

                # Outputs are equivalently long to inputs
                truncate_output = False
                if len(input_token) + len(output_token) + offset > max_len:
                    nc += 1
                    if not remove:
                        if task.startswith("inst:piqa") or task.startswith(
                                "inst:yahoo_answers_topics"):
                            input_token = input_token[:max_len // 2]

                            if len(output_token) > max_len // 2 - offset:
                                output_token = output_token[:max_len // 2 - offset]
                                options = [opt[:max_len // 2 - offset] for opt in options]
                                # Assume options have similarly long length
                                truncate_output = True

                        elif task.startswith("inst:") and len(input_token) < len(output_token):
                            output_token = output_token[:max_len - offset - len(input_token)]
                            options = [opt[:max_len - offset - len(input_token)] for opt in options]
                            truncate_output = True
                        else:
                            input_token = input_token[:max_len - offset - len(output_token)]
                    else:
                        continue

                dp_['input'].append(input_token)
                dp_['output'].append(output_token)
                dp_['options'].append(options)
                dp_['length'].append(len(input_token) + len(output_token))
                dp_['truncate_output'].append(truncate_output)
                dp_['idx'].append(idx_cur)
                idx_cur += 1

                for key in dp.keys():
                    if key not in ['input', 'output', 'options', 'length', 'idx']:
                        dp_[key].append(dp[key][i])

            data_dict[task] = dp_
            if nc < total:
                assert max(data_dict[task]['length']) + offset <= max_len

            total_all += total
            total_nc += nc
            if nc > 0 and verbose:
                print(f"{task[:20]:20s}: Truncate {nc:5d}/{total:5d} data")

        remain = total_all - total_nc
        if verbose:
            print(
                f"After truncation, {remain}/{total_all} ({remain/total_all*100:.0f}%) data remained"
            )

        data_dict = {k: v for k, v in data_dict.items() if len(v) > 0}

        return data_dict

    def _merge_trainset(self, data_dict):
        """Merge trainset with multiple tasks to single dataset
        Returns: merged dataset
        """

        merged_dataset = defaultdict(list)
        for task, dp in data_dict.items():
            for key in dp.keys():
                merged_dataset[key] += dp[key]

        n = len(merged_dataset['input'])
        assert n == len(merged_dataset['output'])
        assert n == len(merged_dataset['options'])

        print(f"Train set merged: {n} data")
        return merged_dataset

    def _merge_eval_demon_dict(self, eval_demon_dict):
        """Merge trainset with multiple tasks to single dataset
        Returns: merged dataset
        """

        seed_list = list(self.eval_demon_dict.keys())
        task_list = self.eval_demon_dict[seed_list[0]].keys()

        eval_demon_dict_ = {}
        for task in task_list:
            merged_dataset = defaultdict(list)
            for seed in seed_list:
                dp = eval_demon_dict[seed][task]
                for key in dp.keys():
                    merged_dataset[key] += dp[key]

            n = len(merged_dataset['input'])
            assert n == len(merged_dataset['output'])
            assert n == len(merged_dataset['options'])

            eval_demon_dict_[task] = merged_dataset

        return eval_demon_dict_

    def _sample_demonstration(self, data_dict, task, idx, k=16):
        """Sample k demonstrations from data_dict[task]
        Returns: sampled demonstrations
        """
        dataset = data_dict[task]

        n = len(dataset['input'])
        idx_demon = (np.random.choice(n - 1, k, replace=False) + idx + 1) % n

        inputs = [dataset['input'][i] for i in idx_demon]
        outputs = [dataset['output'][i] for i in idx_demon]

        return {'input': inputs, 'output': outputs}

    def _concat_demonstration(self, demons, max_demon_length, sum_token=None, sum_recur=False):
        """Concat multiple demonstrations into single input
        Returns: concated demonstration tokens
        """
        inputs = demons['input']
        outputs = demons['output']

        n = len(inputs)

        token = []
        comp_token_length = 0
        for i in range(n):
            token_ = []

            token_ += self.input_prefix  # can contain bos token
            token_ += inputs[i]
            token_ += self.output_prefix
            token_ += outputs[i]
            if self.sep_special:
                token_ += self.eos_token
                comp_token_length += len(self.eos_token)

            if self.online:
                token_ += self.comp_token
                comp_token_length += len(self.comp_token)

            if sum_recur:
                token_ += sum_token
                comp_token_length += len(sum_token)

            # Ignore number of comp_token
            if len(token) + len(token_) > max_demon_length + comp_token_length:
                break
            else:
                token += token_

        if not self.online:
            token += self.comp_token

        return token

    def demons_from_instance(self,
                             instance,
                             max_demon_length=512,
                             k=16,
                             random_k=False,
                             sum_token=None,
                             sum_recur=False,
                             seed_eval=100):
        """Sample and concat demonstrations from a dataset of the instance
        Returns: concated demonstration tokens
        """
        task = instance['task']
        idx = instance['idx']

        if instance['is_train']:
            if random_k:
                if np.random.uniform() <= 0.5:
                    k = np.random.randint(1, k + 1)

            demons = self._sample_demonstration(self.train_dict, task, idx, k=k)
        else:
            dataset = self.eval_demon_dict[seed_eval][task]
            demons = {'input': dataset['input'][:k], 'output': dataset['output'][:k]}

            if k > len(demons['input']):
                k -= len(demons['input'])

                seed_add = 13
                dataset = self.eval_demon_dict[seed_add][task]
                demons['input'] += dataset['input'][:k]
                demons['output'] += dataset['output'][:k]

        demon_token = self._concat_demonstration(demons,
                                                 max_demon_length=max_demon_length,
                                                 sum_token=sum_token,
                                                 sum_recur=sum_recur)

        return demon_token

    def add_demonstration(self,
                          instance,
                          max_length=1024,
                          k=16,
                          random_k=False,
                          seed_eval=100,
                          sum_token=None,
                          sum_recur=False):
        instance_ = {}

        inputs = self.input_prefix + instance['input'] + self.output_prefix
        outputs = instance['output']

        if k > 0:
            max_demon_length = max_length - len(inputs) - len(outputs)
            demon_token = self.demons_from_instance(instance,
                                                    max_demon_length,
                                                    k=k,
                                                    random_k=random_k,
                                                    sum_token=sum_token,
                                                    sum_recur=sum_recur,
                                                    seed_eval=seed_eval)
        else:
            demon_token = []

        if (sum_token is not None) and not sum_recur:
            demon_token += sum_token

        instance_["input_ids"] = demon_token + inputs
        instance_["output_ids"] = outputs

        if len(self.bos_token) >= 1:
            if self.bos_token[0] != instance_["input_ids"][0]:
                instance_["input_ids"] = self.bos_token + instance_["input_ids"]

        if not instance['truncate_output']:
            instance_["output_ids"] += self.eos_token

        return instance_


if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer, LlamaTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--t5', action='store_true')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--sep_special', action='store_true')
    parser.add_argument('--max_example_length', default=256, type=int)
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--n_tok', default=1, type=int)
    args = parser.parse_args()

    is_llama = not args.t5
    if is_llama:
        tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"<COMP{k}>" for k in range(args.n_tok)]})
    tokenizer.model_max_length = 10000
    tokenizer.comp_token_id = tokenizer.additional_special_tokens_ids

    comp_token = tokenizer.comp_token_id
    meta = MetaICLData(tokenizer,
                       is_llama,
                       max_example_length=256,
                       remove=True,
                       comp_token=comp_token,
                       online=args.online,
                       sep_special=args.sep_special)

    print("\n* Train tasks")
    for task, dp in meta.train_dict.items():
        if (dp['options'][0] == dp['options'][1]) and (len(dp['options'][0]) > 0):
            option_text = [tokenizer.decode(opt) for opt in dp['options'][0]]
        else:
            continue
        print(f"{task:20s}: {option_text}")

    print("\n* Eval tasks")
    for task, dp in meta.eval_dict.items():
        if dp['options'][0] == dp['options'][1]:
            option_text = [tokenizer.decode(opt) for opt in dp['options'][0]]
        else:
            continue
        print(f"{task:20s}: {option_text}")

    analyze_data = False
    if analyze_data:
        analyze_data_length(tokenizer, meta.train_dict, max_length=256)
        analyze_data_length(tokenizer, meta.eval_dict, max_length=256)

        analyze_dataset_size(meta.train_dict)
        analyze_dataset_size(meta.eval_dict)

    random_k = False

    # Check demonstration samples
    show_example = False
    if show_example:
        k = 4
        for i in range(2):
            print("\n* Train sample")
            instance = meta.train_dataset[i]
            instance = meta.add_demonstration(instance, k=k, random_k=random_k)
            print(tokenizer.decode(instance['input_ids']))
            print(tokenizer.decode(instance['output_ids']))

            print("\n* Eval sample")
            data_name = list(meta.eval_dataset.keys())[0]
            instance = meta.eval_dataset[data_name][i]
            instance = meta.add_demonstration(instance, k=k, random_k=random_k)
            print(tokenizer.decode(instance['input_ids']))
            print(tokenizer.decode(instance['output_ids']))

            if meta.eval_dataset_option is not None:
                print("\n* Eval sample with option")
                instance = meta.eval_dataset_option[data_name][i]
                instance = meta.add_demonstration(instance, k=k, random_k=random_k)
                print(tokenizer.decode(instance['input_ids']))
                print(tokenizer.decode(instance['output_ids']))
