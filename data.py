import random

from torch.utils.data import Dataset, Dataset, Sampler
from transformers import AutoTokenizer
import pandas as pd


from config_lm import *


class DoubleSynonymsDataset(Dataset):

    """ Create a datset for the boylm model """

    def __init__(self, csv_file):
        self.csv_file = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(config_lm.model)
        self.csv_file = self.new_masked_columns(self.csv_file)
        self.tokenized_text = self.tokenzing(self.csv_file, self.mode)

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        return self.tokenized_text[idx]

    def mask_target(self, sentence):
        sentence = sentence.split()
        length = len(sentence)
        masked_index = random.randrange(0, length, 1)
        sentence[masked_index] = self.tokenizer.mask_token

        return " ".join(sentence)

    def new_masked_columns(self, file):
        file["source1_masked"] = file["source_1"].apply(self.mask_target)
        file["source2_masked"] = file["source_2"].apply(self.mask_target)

        file["text_masked"] = file.apply(
            lambda x: [
                [x["source1_masked"] + " " + x["rela"] + " " + x[f"target"]],
                [x["source2_masked"] + " " + x["rela"] + " " + x[f"target"]]
            ]
               , axis=1
        )

        return file

    def tokenzing(self, file, mode):
        """
        We need labels1 for the unmasked text 1 since the model will do the computation on the labels 1

        """

        text_unmasked = list(zip(*file["text"].tolist()))
        text_unmasked_1, text_unmasked_2 = text_unmasked[0], text_unmasked[1]

        text_masked = list(zip(*file["text_masked"].tolist()))
        text_masked_1, text_masked_2 = text_masked[0], text_masked[1]

        online_network_labels = self.tokenizer(
            text_unmasked_1, padding=True, truncation=True, return_tensors="pt"
        )
        target_network_labels = self.tokenizer(
            text_unmasked_2, padding=True, truncation=True, return_tensors="pt"
        )

        online_network_inputs = self.tokenizer(
            text_masked_1, padding=True, truncation=True, return_tensors="pt"
        )
        target_network_inputs = self.tokenizer(
            text_masked_2, padding=True, truncation=True, return_tensors="pt"
        )

        online_network_labels['input_ids'][online_network_inputs['input_ids'] != self.tokenizer.mask_token_id] = -100

        target_network_labels['input_ids'][target_network_inputs['input_ids'] != self.tokenizer.mask_token_id] = -100
        
        batch_view1 = list(zip(online_network_inputs, online_network_labels))
        batch_view2 = list(zip(target_network_inputs, target_network_labels))
        return list(
                zip(
                    batch_view1,
                    batch_view2
                    ))
        

