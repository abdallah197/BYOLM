import random

from torch.utils.data import Dataset, Dataset, Sampler
from transformers import AutoTokenizer
import pandas as pd


from config import *


class DoubleSynonymsDataset(Dataset):

    """ Create a datset for the boylm model """

    def __init__(self, csv_file):
        self.csv_file = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)

        self.tokenized_two_tagets = self.tokenzing(self.csv_file)

    def __len__(self):
        return len(self.tokenized_two_tagets)

    def __getitem__(self, idx):
        return self.tokenized_two_tagets[idx]

    def mask_target(self, sentence):
        sentence = sentence.split()
        length = len(sentence)
        masked_index = random.randrange(0, length, 1)
        sentence[masked_index] = self.tokenizer.mask_token

        return "".join(sentence)

    def new_masked_columns(self, file):
        file["target1_masked"] = file["target1"].apply(self.mask_target)
        file["target2_masked"] = file["target2"].apply(self.mask_target)
        file["text_masked"] = file.apply(
            lambda x: [
                [x["source"] + " " + x["rela"] + " " + x["target1_masked"]],
                [x["source"] + " " + x["rela"] + " " + x["target2_masked"]],
            ],
            axis=1,
        )

        return file

    def tokenzing(self, file):
        """
        We need labels1 for the unmasked text 1 since the model will do the computation on the labels 1

        """

        text_unmasked = list(zip(*file["text"].tolist()))
        text_unmasked_1, text_unmasked_2 = text_unmasked[0], text_unmasked[1]

        text_masked = file["text_masked"].tolist()
        text_masked_1, text_masked_2 = text_masked[0], text_masked[1]

        text_target_1_unmasked = self.tokenizer(
            text_unmasked_1, padding=True, truncation=True, return_tensors="pt"
        )
        text_target_2_unmasked = self.tokenizer(
            text_unmasked_2, padding=True, truncation=True, return_tensors="pt"
        )

        text_target_1_masked = self.tokenizer(
            text_masked_1, padding=True, truncation=True, return_tensors="pt"
        )
        text_target_2_masked = self.tokenizer(
            text_masked_2, padding=True, truncation=True, return_tensors="pt"
        )

        return list(
            zip(
                text_target_1_unmasked,
                text_target_2_unmasked,
                text_target_1_masked,
                text_target_2_masked,
            )
        )
