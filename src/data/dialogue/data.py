""" The original code is created by Jang-Hyun Kim.
    GitHub Repository: https://github.com/snu-mllab/Context-Memory
"""

import os
import torch
import numpy as np
import datasets
from datasets import Dataset, DatasetDict
from collections import defaultdict
from path_config import DATAPATH


def path_name(model_name):
    if "llama-2" in model_name.lower():
        path_base = os.path.join(DATAPATH, f"dialog/llama2-chat")
    elif "llama" in model_name.lower():
        path_base = os.path.join(DATAPATH, f"dialog/llama")
    else:
        raise ValueError(f"Not supported model name: {model_name}")

    return path_base


class DialogueDataset():
    def __init__(self, tokenizer, comp_token=[], online=True, add_comp_token=True):
        self.tokenizer = tokenizer
        self.model_name = tokenizer.name_or_path.split('/')[-1]
        self.path = path_name(self.model_name)
        self.online = online

        self.bos_token = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        self.eos_token = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        self.comp_token = comp_token
        self.sep_token = tokenizer.encode('a\nA:',
                                          add_special_tokens=False)[1:]  # Prevent strip operation

        if not add_comp_token:
            self.comp_token = []

        assert len(self.sep_token) >= 1, "number of sep_token should be larger than 1"

        # if os.path.exists(os.path.join(self.path, f"trainset.pt")):
        #     self.trainset = torch.load(os.path.join(self.path, f"trainset.pt"))
        #     self.valset = torch.load(os.path.join(self.path, f"valset.pt"))
        dataset = datasets.load_dataset("daily_dialog")
        self.trainset = {}
        self.valset = {}

        self.trainset['dialog'] = self._tokenize(dataset['train']['dialog'])
        self.valset['dialog'] = self._tokenize(dataset['validation']['dialog'])
        self.valset['dialog'] += self._tokenize(dataset['test']['dialog'])

        self.trainset['act'] = dataset['train']['act']
        self.valset['act'] = dataset['validation']['act']
        self.valset['act'] += dataset['test']['act']

        os.makedirs(self.path, exist_ok=True)
        torch.save(self.trainset, os.path.join(self.path, f"trainset.pt"))
        torch.save(self.valset, os.path.join(self.path, f"valset.pt"))

        self.max_len = 400
        self.trainset = self._threshold(self.trainset, max_len_total=self.max_len)
        self.valset = self._threshold(self.valset, max_len_total=600)

        self.trainset['is_train'] = [True] * len(self.trainset['dialog'])
        self.valset['is_train'] = [False] * len(self.valset['dialog'])

        self.train_dataset = Dataset.from_dict(self.trainset, split='train')
        self.eval_dataset = DatasetDict({
            f'turn_{k}': Dataset.from_dict(self._subsample(self.valset, n_turn=k),
                                           split=f'eval_{k-2}')
            for k in [i + 2 for i in [1, 2, 4, 8, 12]]
        })

        print(self.train_dataset)
        print(self.eval_dataset)

    def _preprocess(self, text):
        text = text.replace("。", ".").replace("’", "'")
        text = text.replace(' ,', ',').replace(' .', '.').replace(" '", "'")
        text = text.replace(' ?', '?').replace(' !', '!').replace(' ;', ';')
        text = text.replace("' ", "'")

        text = [t.strip() for t in text.split('.')]
        text = ". ".join(text).strip()

        return text

    def _preprocess_data(self, dialog):
        dialog = [self._preprocess(d) for d in dialog]

        return dialog

    def _tokenize(self, dataset_dialog):
        tokenized_data = []
        for dialog in dataset_dialog:
            dialog = self._preprocess_data(dialog)
            dialog_tokenized = [
                self.tokenizer(d.strip(), add_special_tokens=False)['input_ids'] for d in dialog
            ]
            tokenized_data.append(dialog_tokenized)

        return tokenized_data

    def _threshold(self, dataset, max_len_turn=128, max_len_total=512):
        dataset_ = defaultdict(list)

        for i, dialog in enumerate(dataset['dialog']):
            dialog_token = 0
            skip = False

            # Skip dialog with less than 3 turns
            if len(dialog) <= 2:
                continue

            for turn in dialog:
                dialog_token += len(turn)
                if len(turn) > max_len_turn:
                    skip = True
            if skip:
                continue

            if dialog_token > max_len_total:
                continue

            if not skip:
                for key in dataset.keys():
                    dataset_[key].append(dataset[key][i])

        return dataset_

    def _subsample(self, dataset, n_turn=4):
        dataset_ = defaultdict(list)

        if n_turn == 10:
            n_turn_list = [10]
            # n_turn_list = [9, 10, 11]
        elif n_turn == 14:
            n_turn_list = [15]  # This results in smaller variance than turn 14
            # n_turn_list = [12, 13, 14, 15, 16] # This have similar result patterns to the above setting, n_turn=15
        else:
            n_turn_list = [n_turn]

        for i, dialog in enumerate(dataset['dialog']):
            for n_turn in n_turn_list:
                if len(dialog) >= n_turn:
                    for key in dataset.keys():
                        if type(dataset[key][i]) == list:
                            dataset_[key].append(dataset[key][i][:n_turn])
                        else:
                            dataset_[key].append(dataset[key][i])
        return dataset_

    def _stat(self, dataset):
        n_turn = 0
        n_token = 0

        max_turn = 0
        max_token = 0
        max_dialog_token = 0

        dataset = dataset['dialog']
        for dialog in dataset:
            n_turn += len(dialog)
            dialog_token = 0
            for turn in dialog:
                dialog_token += len(turn)
                max_token = max(max_token, len(turn))

            max_dialog_token = max(max_dialog_token, dialog_token)
            n_token += dialog_token

            max_turn = max(len(dialog), max_turn)

        n = len(dataset)
        print(f"n_turn: {n_turn/n:.0f}, n_data: {n}, n_token: {n_token/n:.0f}, "
              f"max_turn: {max_turn}, max_token: {max_token}, max_dialog_token: {max_dialog_token}")

    def _concat_dialog(self, dialog, sum_token=None, sum_recur=False):
        """Concat multiple dialog into single input
        Returns: concated dialog tokens
        """
        n = len(dialog)
        context = []

        for i in range(n):
            token_ = dialog[i]

            if self.online:
                token_ += self.comp_token

            if sum_recur:
                token_ += sum_token

            context += token_ + self.sep_token

        context = context[:-len(self.sep_token)]  # remove the last sep_token
        if not self.online:
            context += self.comp_token

        if (sum_token is not None) and not sum_recur:
            context += sum_token

        context += self.sep_token
        return context

    def sample_dialog(
        self,
        instance,
        random_k=False,
        sum_token=None,
        sum_recur=False,
        neg_control=False,
    ):
        instance_ = {}

        dialog = instance['dialog']

        if random_k and instance['is_train']:
            k = np.random.randint(3, len(dialog) + 1)
            dialog = dialog[:k]

        context = self._concat_dialog(dialog[:-2], sum_token=sum_token, sum_recur=sum_recur)
        context += dialog[-2] + self.sep_token

        if neg_control:
            context = dialog[-2] + self.sep_token
            # context = dialog[-3] + self.sep_token + dialog[-2] + self.sep_token

        output = dialog[-1]

        instance_["input_ids"] = self.bos_token + context
        instance_["output_ids"] = output + self.eos_token

        return instance_


if __name__ == "__main__":
    from transformers import LlamaTokenizer
    from path_config import CACHEDIR

    chat = True
    if chat:
        tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf',
                                                   cache_dir=CACHEDIR)
    else:
        tokenizer = LlamaTokenizer.from_pretrained('/storage/shared/janghyun/llama-7b-hf')

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    dialog = DialogueDataset(tokenizer, online=False, add_comp_token=False)
