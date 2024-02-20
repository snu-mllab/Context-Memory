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
    if "llama" in model_name.lower():
        path_base = os.path.join(DATAPATH, f"soda/llama")
    elif "mistral" in model_name.lower():
        path_base = os.path.join(DATAPATH, f"dialog/mistral")
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

        print("Loading and processing data... This process takes about a minute.")
        if os.path.exists(os.path.join(self.path, f"trainset.pt")):
            self.trainset = torch.load(os.path.join(self.path, f"trainset.pt"))
            self.valset = torch.load(os.path.join(self.path, f"valset.pt"))
        else:
            dataset = datasets.load_dataset("allenai/soda")
            self.trainset = {}
            self.valset = {}

            print("Tokenizing data...")
            self.trainset['dialog'] = self._tokenize(dataset['train']['dialogue'])
            self.valset['dialog'] = self._tokenize(dataset['validation']['dialogue'])
            # self.valset['dialog'] += self._tokenize(dataset['test']['dialogue'])

            os.makedirs(self.path, exist_ok=True)
            torch.save(self.trainset, os.path.join(self.path, f"trainset.pt"))
            torch.save(self.valset, os.path.join(self.path, f"valset.pt"))

        self.max_len = 512
        self.trainset = self._threshold(self.trainset, max_len_total=self.max_len)
        self.valset = self._threshold(self.valset, max_len_total=self.max_len, n=2000)

        self.trainset['is_train'] = [True] * len(self.trainset['dialog'])
        self.valset['is_train'] = [False] * len(self.valset['dialog'])

        self.train_dataset = Dataset.from_dict(self.trainset, split='train')
        self.eval_dataset = DatasetDict({
            f'turn_{k}':
                Dataset.from_dict(self._subsample(self.valset, n_turn=k), split=f'eval_{k}')
            for k in [3, 6, 9]
        })

        print(self.train_dataset)
        print(self.eval_dataset)

    def _preprocess(self, text):
        text = text.replace("。", ".").replace("’", "'")
        text = text.replace(' ,', ',').replace(' .', '.')
        text = text.replace(' ?', '?').replace(' !', '!').replace(' ;', ';')

        # text = [t.strip() for t in text.split('.')]
        # text = ". ".join(text).strip()

        return text

    def _preprocess_data(self, dialog):
        dialog = [self._preprocess(d) for d in dialog]

        return dialog

    def _tokenize(self, dataset_dialog):
        tokenized_data = []
        for i, dialog in enumerate(dataset_dialog):
            dialog = self._preprocess_data(dialog)
            dialog_tokenized = [
                self.tokenizer(d.strip(), add_special_tokens=False)['input_ids'] for d in dialog
            ]
            tokenized_data.append(dialog_tokenized)

            if i % 10000 == 0:
                print(f"Tokenizing {i}/{len(dataset_dialog)}", end='\r')

        return tokenized_data

    def _threshold(self, dataset, max_len_turn=128, max_len_total=512, n=-1):
        dataset_ = defaultdict(list)
        count = 0

        for i, dialog in enumerate(dataset['dialog']):
            dialog_token = 0
            skip = False

            # Skip dialog with less than 3 turns
            if len(dialog) <= 2:
                count += 1
                continue

            for turn in dialog:
                dialog_token += len(turn)
                if (len(turn) > max_len_turn) or (len(turn) == 0):
                    skip = True
                    count += 1
                    break
            if skip:
                continue

            if dialog_token > max_len_total:
                count += 1
                continue

            if not skip:
                for key in dataset.keys():
                    dataset_[key].append(dataset[key][i])

                if n > 0 and len(dataset_['dialog']) >= n:
                    break

        print(f"Skip {count} dialogues")
        return dataset_

    def _subsample(self, dataset, n_turn=4):
        dataset_ = defaultdict(list)

        for i, dialog in enumerate(dataset['dialog']):
            if len(dialog) >= n_turn and len(dialog[n_turn - 1]) > 0:
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

    tokenizer.add_special_tokens({"additional_special_tokens": [f"<COMP{k}>" for k in range(2)]})
    tokenizer.comp_token_id = tokenizer.additional_special_tokens_ids[:1]
    tokenizer.sum_token_id = tokenizer.additional_special_tokens_ids[1:]

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    dialog = DialogueDataset(tokenizer, comp_token=tokenizer.comp_token_id, online=True)

    for i in range(1):
        print("Train set")
        for _ in range(3):
            instance = dialog.train_dataset[i]
            token = dialog.sample_dialog(instance, sum_token=tokenizer.sum_token_id, random_k=True)
            print(tokenizer.decode(token["input_ids"]))
            print(tokenizer.decode(token["output_ids"]))
            print()

    for k, dataset in dialog.eval_dataset.items():
        print(k)
        instance = dataset[0]
        token = dialog.sample_dialog(instance, sum_token=tokenizer.sum_token_id)
        print(tokenizer.decode(token["input_ids"]))
        print(tokenizer.decode(token["output_ids"]))
        print()

    dialog._stat(dialog.trainset)
    dialog._stat(dialog.valset)

    for k, v in dialog.eval_dataset.items():
        dialog._stat(v)
