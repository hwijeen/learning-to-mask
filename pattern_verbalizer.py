from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch


ANSWER_TOKEN = "_"


@dataclass
class DataCollatorForClozeTask(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        labels = torch.empty_like(batch["input_ids"]).fill_(-100)

        answer_token_id = self.tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
        index = (torch.tensor(batch["input_ids"]) == answer_token_id).nonzero()
        batch["input_ids"][range(len(labels)), index[:, 1]] = \
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        labels[range(len(labels)), index[:, 1]] = batch.pop("label")
        batch["labels"] = labels
        return batch


def rte_pv_fn(sent1s, sent2s, labels=None):
    sent1_format = "{}?"
    sent2_format = "{}, {}"
    formatted_sent1s, formatted_sent2s = [], []
    verbalizer = {
            0: "No",
            1: "Yes",
            -1: "dummy",  # test data
            }
    formatted_labels  = []
    for sent1, sent2, label in zip(sent1s, sent2s, labels):
        assert ANSWER_TOKEN not in sent1 and ANSWER_TOKEN not in sent2
        formatted_sent1s.append(sent1_format.format(sent1))
        formatted_sent2s.append(sent2_format.format(ANSWER_TOKEN, sent2))
        formatted_labels.append(verbalizer[label])

    return formatted_sent1s, formatted_sent2s, formatted_labels

#     return ['"', text_b, '" ?'], [self.mask, ', "', text_a, '"']
#         # return [text_b, '?'], [self.mask, ',', text_a]
#         # return ['"', text_b, '" ?'], [self.mask, '. "', text_a, '"']
#         # return [text_b, '?'], [self.mask, '.', text_a]
#         # return [text_a, ' question: ', self.shortenable(example.text_b), ' True or False? answer:', self.mask], []
