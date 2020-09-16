import random
import time

from torch.utils.data import Dataset, Dataset, Sampler
from transformers import AutoTokenizer
import pandas as pd

import pandas as pd
import random
import ast
from functools import reduce
import operator
from torch import Tensor
import collections
import torch

import config_lm


class DoubleSynonymsDataset(Dataset):

    """ Create a datset for the boylm model """

    def __init__(self, csv_file):
        self.csv_file = pd.read_csv(csv_file, nrows = 1000)
        self.tokenizer = AutoTokenizer.from_pretrained(config_lm.models["bert"])
        self.csv_file = self.new_masked_columns(self.csv_file)
        self.tokenized_text = self.tokenzing(self.csv_file)

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        return (self.tokenized_text[0][idx], self.tokenized_text[1][idx])

    def mask_target(self, file):
        sentence = file["target"].split()
        length = len(sentence)
        sentence[file["masked_index"]] = self.tokenizer.mask_token

        return " ".join(sentence)

    def new_masked_columns(self, file):
        file["masked_index"] = file.apply(
            lambda x: random.randrange(0, len(x["target"].split()), 1), axis=1
        )
        file["target"] = file.apply(lambda x: self.mask_target(x), axis=1)
        file["masked_index_1"] = file.apply(
            lambda x: x["masked_index"] + len(x["source_1"].split()), axis=1
        )
        file["masked_index_2"] = file.apply(
            lambda x: x["masked_index"] + len(x["source_2"].split()), axis=1
        )

        file["text_masked"] = file.apply(
            lambda x: [
                [x["source_1"] + " " + x["rela"] + " " + x["target"]],
                [x["source_2"] + " " + x["rela"] + " " + x["target"]],
            ],
            axis=1,
        )

        return file

    def flatten(self, lis):
        return reduce(operator.concat, lis)

    def filtering_ids(self, tokens):
        masked_indexes = collections.defaultdict(int)
        for i, ids in enumerate(tokens["input_ids"]):
            try:
                masked_indexes[i] = ids.tolist().index(103)
            except:
                continue
        consider = masked_indexes.keys()
        mask = [0] * tokens["input_ids"].shape[0]

        for i in masked_indexes:
            mask[i] = 1
        mask = torch.nonzero(torch.tensor(mask))
        tokens["input_ids"] = tokens["input_ids"][mask].squeeze()
        tokens["token_type_ids"] = tokens["token_type_ids"][mask].squeeze()
        tokens["attention_mask"] = tokens["attention_mask"][mask].squeeze()
        return tokens, list(masked_indexes.values())

    def tokenzing(self, file):
        """
        We need labels1 for the unmasked text 1 since the model will do the computation on the labels 1
        """
        file.text = file.text.apply(lambda s: list(ast.literal_eval(s)))

        text_unmasked = list(zip(*file["text"].tolist()))

        text_unmasked_1, text_unmasked_2 = (
            self.flatten(list(text_unmasked[0])),
            self.flatten(list(text_unmasked[1])),
        )

        text_masked = list(zip(*file["text_masked"].tolist()))
        text_masked_1, text_masked_2 = (
            self.flatten(list(text_masked[0])),
            self.flatten(list(text_masked[1])),
        )

        online_network_labels = self.tokenizer(
            text_unmasked_1,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config_lm.max_lenght,
        )
        target_network_labels = self.tokenizer(
            text_unmasked_2,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config_lm.max_lenght,
        )

        online_network_inputs = self.tokenizer(
            text_masked_1,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config_lm.max_lenght,
        )
        target_network_inputs = self.tokenizer(
            text_masked_2,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config_lm.config_lm.max_lenght,
        )

        online_network_labels["input_ids"][
            online_network_inputs["input_ids"] != self.tokenizer.mask_token_id
        ] = -100

        target_network_labels["input_ids"][
            target_network_inputs["input_ids"] != self.tokenizer.mask_token_id
        ] = -100

        # filtering the tensors to get tensors that have masks along with the masked index as well

        online_network_inputs, masks_o = self.filtering_ids(online_network_inputs)
        target_network_inputs, masks_t = self.filtering_ids(target_network_inputs)

        batch_view1 = list(zip(online_network_inputs["input_ids"], online_network_labels["input_ids"], masks_o))
        batch_view2 = list(zip(target_network_inputs["input_ids"], target_network_labels["input_ids"], masks_t))

        return (batch_view1, batch_view2)

start = time.time()
print("start time: ", start)

f = "/GW/Health-Corpus/work/UMLS/data/data.csv"
batches = DoubleSynonymsDataset(f)
end = time.time()
print("end time: ", end - start)
