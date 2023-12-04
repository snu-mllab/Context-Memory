import os
import json
import numpy as np

import datasets
from datasets.splits import NamedSplit
from pathlib import Path
from path_config import DATAPATH

logger = datasets.logging.get_logger(__name__)


class LaMPConfig(datasets.BuilderConfig):
    def __init__(
        self,
        max_tok_len=100,
        cutoff_ratio=0.9,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        dataset_name = kwargs.get("name")
        # dataset_name: lamp${i} | lamp${i}_time

        lamp_index = int(dataset_name[4])
        time_based = dataset_name.endswith("time")
        self.base_dir = Path(os.path.join(DATAPATH, f"lamp/lamp{lamp_index}"))

        if time_based:
            self.train_questions = self.base_dir / f"train_questions_time_lamp{lamp_index}.json"
            self.train_outputs = self.base_dir / f"train_outputs_time_lamp{lamp_index}.json"
            self.validation_questions = self.base_dir / f"dev_questions_time_lamp{lamp_index}.json"
            self.validation_outputs = self.base_dir / f"dev_outputs_time_lamp{lamp_index}.json"
            self.test_questions = self.base_dir / f"test_questions_time_lamp{lamp_index}.json"
        else:
            self.train_questions = self.base_dir / f"train_questions_lamp{lamp_index}.json"
            self.train_outputs = self.base_dir / f"train_outputs_lamp{lamp_index}.json"
            self.validation_questions = self.base_dir / f"dev_questions_lamp{lamp_index}.json"
            self.validation_outputs = self.base_dir / f"dev_outputs_lamp{lamp_index}.json"
            self.test_questions = self.base_dir / f"test_questions_lamp{lamp_index}.json"

        self.time_based = time_based
        self.lamp_index = lamp_index
        self.max_tok_len = max_tok_len
        if lamp_index == 3:
            self.max_tok_len = 128
        self.cutoff_ratio = cutoff_ratio

    def gen_kwargs(self, split):

        # tokenizer = "t5" if self.is_t5 else "llama"
        tokenizer = "t5"

        if split == "train":
            questions_path = self.train_questions
            outputs_path = self.train_outputs
        elif split == "validation":
            questions_path = self.validation_questions
            outputs_path = self.validation_outputs
        elif split == "test":
            questions_path = self.test_questions
            outputs_path = None
        else:
            raise ValueError("Invalid Split!")

        split_ = "dev" if split == "validation" else split
        split_based = "time" if self.time_based else "user"
        tok_len_path = self.base_dir / f"lamp{self.lamp_index}_ppep_token_len_{split_}_{split_based}_{tokenizer}.npy"
        # input_len_path = self.base_dir / f"lamp{self.lamp_index}_profile_entry_len_{split_}_{split_based}_{tokenizer}.npy"

        # if split != "test" and self.lamp_index > 3:
        #     output_len_path = self.base_dir / f"lamp{self.lamp_index}_output_token_len_{split_}_{split_based}_{tokenizer}.npy"
        # else:
        #     output_len_path = None

        gen_kwargs = {
            "questions_path": questions_path,
            "outputs_path": outputs_path,
            "tok_len_path": tok_len_path,
            "lamp_index": self.lamp_index,
            "max_tok_len": self.max_tok_len,
            "cutoff_ratio": self.cutoff_ratio,
            "split": split,
        }
        return gen_kwargs


class LaMP(datasets.GeneratorBasedBuilder):
    """LaMP Dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = LaMPConfig

    def _info(self):

        profile_feature_dict = {"id": datasets.Value("string")}
        lamp_index = self.config.lamp_index
        time_based = self.config.time_based
        stringtype = datasets.Value("string")
        description = f"LaMP{lamp_index}(time_based) Data" if time_based \
                        else f"LaMP{lamp_index} Data"

        if time_based:
            profile_feature_dict.update({"date": stringtype})

        profile_features = {}
        if lamp_index == 1:
            profile_features = {
                "title": stringtype,
                "abstract": stringtype,
            }
        elif lamp_index == 2:
            profile_features = {
                "title": stringtype,
                "category": stringtype,
                "text": stringtype,
            }
        elif lamp_index == 3:
            profile_features = {
                "text": stringtype,
                "score": stringtype,
            }
        elif lamp_index == 4:
            profile_features = {
                "text": stringtype,
                "title": stringtype,
            }
        elif lamp_index == 5:
            profile_features = {"title": stringtype, "abstract": stringtype}
        elif lamp_index == 7:
            profile_features = {
                "text": stringtype,
            }
        profile_feature_dict.update(profile_features)

        return datasets.DatasetInfo(
            description=description,
            features=datasets.Features({
                "input": stringtype,
                "profile": datasets.features.Sequence(profile_feature_dict),
                "output": stringtype,
                "split": stringtype,
                "lamp_index": datasets.Value("int8"),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        del dl_manager
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=self.config.gen_kwargs("train"),
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation"),
                gen_kwargs=self.config.gen_kwargs("validation"),
            ),
            datasets.SplitGenerator(
                name=NamedSplit("test"),
                gen_kwargs=self.config.gen_kwargs("test"),
            ),
        ]

    def _generate_examples(
        self,
        questions_path,
        outputs_path,
        tok_len_path,
        lamp_index,
        max_tok_len,
        cutoff_ratio,
        split,
    ):
        """Yields examples."""

        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        if outputs_path is not None and split != "test":
            with open(outputs_path, "r", encoding="utf-8") as f:
                outputs = json.load(f)
            outputs = outputs["golds"]
        else:
            outputs = None

        token_len_matrix = np.load(tok_len_path)

        # max_tok_len >= (max_tok_for_entry * num_shot + ", and " * (num_shot - 1)) + "Input: " + "Output: " + "Instruction: " + input_len
        # -> approximately max_tok_len = num_shot * (max_tok_for_entry + 4) + 10 + input_len
        # margin_prompt = 10
        # margin_conjuction = 4
        # max_tok_for_entry = (max_tok_len - margin_prompt - input_len) // num_shot - margin_conjuction if num_shot > 0 else np.ones_like(input_len) * np.inf

        # Compute only once
        valid_entries = token_len_matrix > 0  # mask for valid entry
        cutoff = (token_len_matrix < max_tok_len) * valid_entries
        remains = cutoff.sum(axis=1)
        total = valid_entries.sum(axis=1)
        ratio = remains / total  # ratio[i] == profile remaining ratio of ith datapoint after cutoff

        if lamp_index != 3:
            print(f"max_tok_len={max_tok_len}, cutoff_ratio={cutoff_ratio}")
            print(f"{(ratio < cutoff_ratio).sum() / ratio.size * 100:.2f}% removed")
        else:
            print(f"max_tok_len={max_tok_len}, cutoff_value=32")
            print(f"{(remains < 32).sum() / remains.size * 100:.2f}% removed")

        if outputs is not None:
            for i, (question, output) in enumerate(zip(questions, outputs)):
                id_ = question["id"]
                profile = question["profile"]
                if "date" in profile and isinstance(profile["date"], int):
                    profile["date"] = str(profile["date"])

                if lamp_index != 3 and ratio[i] < cutoff_ratio:
                    continue
                elif lamp_index == 3 and remains[i] < 32:
                    continue

                profile = [p for j, p in enumerate(profile) if cutoff[i][j]]

                yield id_, {
                    "input": question["input"],
                    "profile": profile,
                    "output": output["output"],
                    "split": split,
                    "lamp_index": self.config.lamp_index,
                }
        else:
            for i, question in enumerate(questions):
                id_ = question["id"]
                profile = question["profile"]
                if "date" in profile and isinstance(profile["date"], int):
                    profile["date"] = str(profile["date"])

                if lamp_index != 3 and ratio[i] < cutoff_ratio:
                    continue
                elif lamp_index == 3 and remains[i] < 32:
                    continue

                profile = [p for j, p in enumerate(profile) if cutoff[i][j]]

                yield id_, {
                    "input": question["input"],
                    "profile": profile,
                    "output": "",
                    "split": split,
                    "lamp_index": self.config.lamp_index,
                }
