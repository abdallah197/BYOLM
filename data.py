import random
import time

from torch.utils.data import Dataset, Dataset, Sampler, ConcatDataset

from transformers import AutoTokenizer
import pandas as pd

import random
import ast
from functools import reduce
import operator
from torch import Tensor
import collections
import torch

import config_lm


def mask_target(file, tokenizer):
    sentence = file["target"].split()
    sentence[file["masked_index"]] = tokenizer.mask_token
    return " ".join(sentence)


def flatten(lis):
    return reduce(operator.concat, lis)


def preprocesser(file):
    file["masked_index"] = file.apply(
        lambda x: random.randrange(0, len(x["target"].split()), 1), axis=1
    )
    file["target"] = file.apply(
        lambda x: self.mask_target(x, tokenizer=self.tokenizer), axis=1
    )
    file["masked_index_1"] = file.apply(
        lambda x: x["masked_index"] + len(x["source_1"].split()), axis=1
    )
    file["masked_index_2"] = file.apply(
        lambda x: x["masked_index"] + len(x["source_2"].split()), axis=1
    )
    index_names = file[(file["source_1"] == file["source_2"])].index
    file.drop(index_names, inplace=True)
    l1 = file.apply(
        lambda x: x["source_1"] + " " + x["rela"] + " " + x["target"], axis=1
    ).tolist()
    l2 = file.apply(
        lambda x: x["source_2"] + " " + x["rela"] + " " + x["target"], axis=1
    ).tolist()

    return l1, l2


class DoubleSynonymsDataset(Dataset):
    """ Create a datset for the boylm model """

    def __init__(self, triples):
        self.tokenizer = AutoTokenizer.from_pretrained(config_lm.models["bert"])
        self.list_of_tokens = self.tokenzing(triples)

    def __len__(self):
        return len(self.list_of_tokens)

    def __getitem__(self, idx):

        return self.list_of_tokens[idx]

    def filtering_ids(self, tokens, msk=None):
        masked_indexes = collections.defaultdict(int)
        for i, ids in enumerate(tokens["input_ids"]):
            try:
                masked_indexes[i] = ids.tolist().index(103)
            except:
                continue
        # to deduct the same exaxmples from both datasets
        if torch.is_tensor(msk):
            mask = msk
        else:
            mask = [0] * tokens["input_ids"].shape[0]
            for i in masked_indexes:
                mask[i] = 1
            mask = torch.nonzero(torch.tensor(mask))
        tokens["input_ids"] = tokens["input_ids"][mask].squeeze()
        tokens["token_type_ids"] = tokens["token_type_ids"][mask].squeeze()
        tokens["attention_mask"] = tokens["attention_mask"][mask].squeeze()
        return tokens, list(masked_indexes.values()), mask

    def tokenzing(self, list_of_tokens):
        """
        We need labels1 for the unmasked text 1 since the model will do the computation on the labels 1
        """

        network_inputs = self.tokenizer(
            list_of_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config_lm.max_length,
        )

        # filtering the tensors to get tensors that have masks along with the masked index as well
        network_inputs, masks_o, include_exmp = self.filtering_ids(network_inputs)
        view = list(zip(network_inputs["input_ids"], masks_o))
        return view


class ConcatDataset(ConcatDataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return list(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
