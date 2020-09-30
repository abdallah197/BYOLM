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

class Preprocesser:
    
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(config_lm.models["bert"])
        self.df = self.new_masked_columns(self.df)

    
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
    
class DoubleSynonymsDataset(Dataset):

    """ Create a datset for the boylm model """

    def __init__(self, csv_file, dataset):
        self.dataset = dataset
        self.csv_file = csv_file
        self.tokenizer = AutoTokenizer.from_pretrained(config_lm.models["bert"])
        self.tokenized_text = self.tokenzing(self.csv_file, self.dataset)

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):

        batch = self.tokenized_text
        return batch[idx]

    def flatten(self, lis):
        return reduce(operator.concat, lis)

    def filtering_ids(self, tokens, msk = None):
        masked_indexes = collections.defaultdict(int)
        for i, ids in enumerate(tokens["input_ids"]):
            try:
                masked_indexes[i] = ids.tolist().index(103)
            except:
                continue
        consider = masked_indexes.keys()
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

    def tokenzing(self, file, dataset):
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
        online_network_inputs = self.tokenizer(
                text_masked_1,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=config_lm.max_length,
            )
        
        # filtering the tensors to get tensors that have masks along with the masked index as well
        online_network_inputs , masks_o, include_exmp = self.filtering_ids(online_network_inputs)
        
        target_network_inputs = self.tokenizer(
                text_masked_2,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=config_lm.max_length,
            )
        target_network_inputs, masks_t, include_exmp = self.filtering_ids(target_network_inputs, include_exmp)

        if dataset == "online":
            online_network_labels = self.tokenizer(
                text_unmasked_1,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=config_lm.max_length,
            )
            
            online_network_labels ,_ , _ = self.filtering_ids(online_network_labels, include_exmp)
            online_network_labels["input_ids"][
                online_network_inputs["input_ids"] != self.tokenizer.mask_token_id
            ] = -100
            
            # TODO: switch onlines back to target_network_inputs
            

            batch_view = list(
                zip(
                    online_network_inputs["input_ids"],
                    online_network_inputs["token_type_ids"],
                    online_network_inputs["attention_mask"],
                    online_network_labels["input_ids"],
                    masks_o,
                )
            )


        else:
            target_network_labels = self.tokenizer(
                text_unmasked_2,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=config_lm.max_length,
            )
            target_network_labels ,_ , _ = self.filtering_ids(target_network_labels, include_exmp)
            target_network_labels["input_ids"][
                target_network_inputs["input_ids"] != self.tokenizer.mask_token_id
            ] = -100
            # TODO: switxh targets back to target_network_inputs
            batch_view = list(
                zip(
                    target_network_inputs["input_ids"],
                    target_network_inputs["token_type_ids"],
                    target_network_inputs["attention_mask"],
                    target_network_labels["input_ids"],
                    masks_t,
                )
            )

        return batch_view

class ConcatDataset(ConcatDataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return list(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

# start = time.time()
# print("start time: ", start)

# f = "/GW/Health-Corpus/work/UMLS/data/data.csv"
# batches = DoubleSynonymsDataset(f)
# end = time.time()
# print("end time: ", end - start)
