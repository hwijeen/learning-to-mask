#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, Value
import evaluate
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.integrations import TensorBoardCallback

from layers import MaskedLinear, MaskedEmbedding
from utils import recursive_setattr, calculate_sparsity, chain, get_mask, calculate_hamming_dist
from pattern_verbalizer import rte_pv_fn, sst2_pv_fn, cola_pv_fn, qqp_pv_fn, qnli_pv_fn, mnli_pv_fn_2, DataCollatorForClozeTask, ANSWER_TOKEN
from fisher import *


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

glue_tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_pv_fn = {
    "rte" : rte_pv_fn,
    "mrpc" : rte_pv_fn,  # FIXME: for now
    "sst2" : sst2_pv_fn,
    "imdb" : sst2_pv_fn,  # FIXME: for now
    "yelp_polarity" : sst2_pv_fn,  # FIXME: for now
    "amazon_polarity" : sst2_pv_fn,  # FIXME: for now
    "cola" : cola_pv_fn,
    "qqp" : qqp_pv_fn,
    "qnli": qnli_pv_fn,
    "mnli": mnli_pv_fn_2,
    "hans" : rte_pv_fn,
    "sick" : mnli_pv_fn_2,
    "snli" : mnli_pv_fn_2,
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    cloze_task: Optional[bool] = field(default=False, metadata={"help": "Formulate downstream task as cloze task"})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    # TODO: make this a standalone argument
    initial_sparsity: float = field(default=0.0, metadata={"help": "Initial sparsity."})
    init_scale: float = field(default=0.02, metadata={"help": "Initial scale."})
    threshold: float = field(default=0.01, metadata={"help": "Threshold."})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"help": "Attention probs dropout prob."})
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "Hidden dropout prob."})


@dataclass
class SparseUpdateTrainingArguments(TrainingArguments):
    num_samples: int = field(
        # default=1024,
        default=0,
        metadata={"help": "The number of samples to compute parameters importance"}
    )
    # keep_ratio: float = field(
    #     # default=0.005,
    #     default=0.95,
    #     metadata={"help": "The trainable parameters to total parameters."}
    # )
    mask_method: str = field(
        default="label-absolute",
        metadata={"help": "The method to select trainable parameters. Format: sample_type-grad_type, \
                   where sample_type in \{label, expect\}, and grad_type in \{absolute, square\}"}
    )
    normal_training: bool = field(
        default=False,
        metadata={"help": "Whether to use typical BERT training method."}
    )
    mask_path: str = field(
        default="",
        metadata={"help": "The path for existing mask."}
    )
    data_split_path: str = field(
        default="",
        metadata={"help": "The path for existing training data indices."}
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SparseUpdateTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        if data_args.dataset_name in glue_tasks:
            task_data = ("glue", data_args.dataset_name)
        else:
            task_data = (data_args.dataset_name,)

        raw_datasets = load_dataset(
            *task_data,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if data_args.dataset_name in ["amazon_polarity"]:
            raw_datasets["validation"] = raw_datasets["test"]
            raw_datasets.pop("test")
        if data_args.dataset_name in ["imdb", "yelp_polarity"]:
            raw_datasets["test"].task_templates.pop()
            raw_datasets["train"].task_templates.pop()
            raw_datasets["validation"] = raw_datasets["test"]
            raw_datasets.pop("test")
            if data_args.dataset_name == "imdb":
                raw_datasets.pop("unsupervised")
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.dataset_name is not None:
        is_regression = data_args.dataset_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    elif data_args.dataset_name in ["imdb", "yelp_polarity", "amazon_polarity" ]:
        is_regression = False
        num_labels = 2

    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    if not training_args.do_train and training_args.do_eval:
        raw_datasets["validation"] = datasets.concatenate_datasets([raw_datasets["train"], raw_datasets["validation"]])

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.dataset_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.cloze_task:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    # if model_args.initial_sparsity != 0.0:
    #     for n, p in model.named_parameters():
    #         p.requires_grad = False
    #     for n, m in model.named_modules():
    #         if isinstance(m, nn.Linear):
    #             masked_linear = MaskedLinear(m.weight,
    #                                          m.bias,
    #                                          mask_scale=model_args.init_scale,
    #                                          threshold=model_args.threshold,
    #                                          initial_sparsity=model_args.initial_sparsity,
    #                                          )
    #             masked_linear.mask_real.requires_grad = True
    #             masked_linear.bias.requires_grad = False
    #             recursive_setattr(model, n, masked_linear)
    #         # elif isinstance(m, nn.Embedding):
    #         #     masked_embedding = MaskedEmbedding(m.weight,
    #         #                                        m.padding_idx,
    #         #                                        mask_scale=model_args.init_scale,
    #         #                                        threshold=model_args.threshold,
    #         #                                        initial_sparsity=model_args.initial_sparsity
    #         #                                        )
    #         #     masked_embedding.mask_real.requires_grad = True
    #         #     recursive_setattr(model, n, masked_embedding)
    #     print(f"\n\n ========== Initial sparsity: {calculate_sparsity(model)} ==========\n\n")

    if os.path.isdir(model_args.model_name_or_path):  # load from saved
        print("Loading from saved model: ", model_args.model_name_or_path)
        state_dict = torch.load(os.path.join(model_args.model_name_or_path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
    # Preprocessing the raw_datasets
    if data_args.dataset_name in glue_tasks:
        sentence1_key, sentence2_key = task_to_keys[data_args.dataset_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    # if (
    #     model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    #     and data_args.task_name is not None
    #     and not is_regression
    # ):
    #     # Some have all caps in their config, some don't.
    #     label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    #     if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
    #         label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
    #             "\nIgnoring the model labels as a result.",
    #         )
    # elif data_args.task_name is None and not is_regression:
    #     label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.dataset_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        if data_args.cloze_task:
            result["label"] = tokenizer.convert_tokens_to_ids(examples["label"])
        return result

    if data_args.cloze_task:
        tokenizer.add_special_tokens({"additional_special_tokens": [ANSWER_TOKEN]})
        pv_fn = task_to_pv_fn[data_args.dataset_name]
        def pattern_verbalizer(examples):
            # turn examples into pvp format, depending on the task
            sent1s = examples[sentence1_key]
            sent2s = examples[sentence2_key] if sentence2_key is not None else None
            labels = examples["label"]
            sent1s, sent2s, labels = pv_fn(sent1s, sent2s, labels)
            examples[sentence1_key] = sent1s
            if sentence2_key is not None:
                examples[sentence2_key] = sent2s
            examples["label"] = labels
            return examples
        preprocess_function = chain(pattern_verbalizer, preprocess_function)
        raw_datasets["train"].features["label"] = Value(dtype='int32', id=None)
        if data_args.dataset_name == "mnli":
            raw_datasets["validation_matched"].features["label"] = Value(dtype='int32', id=None)
            raw_datasets["validation_mismatched"].features["label"] = Value(dtype='int32', id=None)
        else:
            raw_datasets["validation"].features["label"] = Value(dtype='int32', id=None)
        if "test" in raw_datasets:
            raw_datasets["test"].features["label"] = Value(dtype='int32', id=None)
        if data_args.dataset_name == "mnli":  # mnli always has test
            raw_datasets["test_matched"].features["label"] = Value(dtype='int32', id=None)
            raw_datasets["test_mismatched"].features["label"] = Value(dtype='int32', id=None)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    #  if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.dataset_name in glue_tasks:
        metric = evaluate.load("glue", data_args.dataset_name)
    else:
        metric = evaluate.load("accuracy")
    #FIXME: in order to calculate F1, label needs to be 0 or 1
    if data_args.dataset_name in ["mrpc", "qqp", "mnli"]:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    if data_args.cloze_task:
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)
    else:
        preprocess_logits_for_metrics = None
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if data_args.dataset_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    if data_args.cloze_task:
        data_collator = DataCollatorForClozeTask(tokenizer)

    # Fisher mask
    fisher_mask = None
    if training_args.num_samples != 0:
        keep_ratio = 1.0 - model_args.initial_sparsity
        if training_args.mask_path != "":
            mask = torch.load(training_args.mask_path, map_location="cpu")
        else:
            if training_args.mask_method == "bias":
                mask_method = create_mask_bias
                mask = create_mask_bias(
                    model, train_dataset, data_collator, training_args.num_samples, keep_ratio
                )

            elif training_args.mask_method == "random":
                mask_method = create_mask_random

                mask = create_mask_random(
                    model, train_dataset, data_collator, training_args.num_samples, keep_ratio
                )

            else:
                sample_type, grad_type = training_args.mask_method.split("-")

                import inspect
                signature = inspect.signature(model.forward)
                signature_columns = list(signature.parameters.keys())
                signature_columns += list(set(["label", "label_ids"]))
                to_remove = list(set(train_dataset.column_names) - set(signature_columns))
                train_dataset = train_dataset.remove_columns(to_remove)
                # train_dataset.set_format(columns=["input_ids", "label"])
                fisher_mask = create_mask_gradient(
                    model,
                    train_dataset,
                    data_collator,
                    training_args.num_samples,
                    keep_ratio,
                    sample_type,
                    grad_type
                )
                print(f"\n\nFisher mask sparsity: {calculate_sparsity(fisher_mask)}\n\n")

    if model_args.initial_sparsity != 0.0:
        for n, p in model.named_parameters():
            p.requires_grad = False
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                mask = None
                if fisher_mask is not None:
                    if n == "cls.predictions.decoder":
                        mask = fisher_mask["bert.embeddings.word_embeddings.weight"]
                    else:
                        mask = fisher_mask[n + ".weight"]
                masked_linear = MaskedLinear(m.weight,
                                             m.bias,
                                             mask_scale=model_args.init_scale,
                                             threshold=model_args.threshold,
                                             initial_sparsity=model_args.initial_sparsity,
                                             mask=mask,
                                             )
                masked_linear.mask_real.requires_grad = True
                masked_linear.bias.requires_grad = False
                recursive_setattr(model, n, masked_linear)
            # elif isinstance(m, nn.Embedding):
            #     masked_embedding = MaskedEmbedding(m.weight,
            #                                        m.padding_idx,
            #                                        mask_scale=model_args.init_scale,
            #                                        threshold=model_args.threshold,
            #                                        initial_sparsity=model_args.initial_sparsity
            #                                        )
            #     masked_embedding.mask_real.requires_grad = True
            #     recursive_setattr(model, n, masked_embedding)
        print(f"\n\n ========== Initial sparsity: {calculate_sparsity(model)} ==========\n\n")

    #

    ##TODO: have README.md report best accuracy
    class ExtendedTensorBoardCallback(TensorBoardCallback):
        """
        Add custom metric to TensorBoard
        report_to should not be set when using this callback
        """
        def on_train_begin(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            i = 2
            while os.path.isdir(args.logging_dir):
                args.logging_dir = args.logging_dir + f"_{i}"
                i += 1
            self.prev_mask_dict = get_mask(model)
            self.init_mask_dict = get_mask(model)

            super().on_train_begin(args, state, control, **kwargs)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero:
                return

            if self.tb_writer is None:
                self._init_summary_writer(args)

            if self.tb_writer is not None:
                model = kwargs['model']
                mask_dict = get_mask(model)
                sparsity = calculate_sparsity(model)
                dist_from_prev = calculate_hamming_dist(self.prev_mask_dict, mask_dict)
                dist_from_init = calculate_hamming_dist(self.init_mask_dict, mask_dict)
                self.tb_writer.add_scalar("mask/chage_from_prev", dist_from_prev, state.global_step)
                self.tb_writer.add_scalar("mask/change_from_init", dist_from_init, state.global_step)
                self.tb_writer.add_scalar("mask/sparsity", sparsity, state.global_step)
                self.prev_mask_dict = mask_dict
            self.tb_writer.flush()

            super().on_log(args, state, control, logs=logs, **kwargs)

    # Initialize our Trainer
    training_args.report_to = []
    callbacks = []
    if model_args.initial_sparsity != 0.0 and not os.path.isdir(model_args.model_name_or_path):
        callbacks = [ExtendedTensorBoardCallback()]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    if model_args.initial_sparsity != 0.0:
        print(f"\n\n ========== Final sparsity: {calculate_sparsity(model)} ==========\n\n")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.dataset_name]
        eval_datasets = [eval_dataset]
        if data_args.dataset_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.dataset_name]
        predict_datasets = [predict_dataset]
        if data_args.dataset_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.dataset_name
        kwargs["dataset"] = f"GLUE {data_args.dataset_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
