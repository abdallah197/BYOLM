import torch
import pandas as pd
import torch.nn as nn
import numpy as np

import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from data import DoubleSynonymsDataset
from model import BYOLLM

import config_lm


class ByolLM:
    """
    main training class
    """

    def __init__(self, online_network, target_network, optimizer, device):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = config_lm.epochs
        self.writer = SummaryWriter()
        self.m = config_lm.m
        self.batch_size = config_lm.train_batch_size
        self.num_workers = config_lm.num_workers
        self.checkpoint_interval = config_lm.checkpoint_interval

    def update_moving_average(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def boyl_loss(self, x, y):
        """
        minimize the loss function 
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True,
        )

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            for (batch_view_1, batch_view_2) in train_loader:
                """
                batch_view_1:
                1. online_network_inputs (online network input ids with a mask
                token in the target)
                2. online_network_labels (same as the online network, includes
                the original token of the masked token
                with all other tokens masked as -100, this vector to beu used when
                using cross entropy loss (supervised learning))
                3. masks_o: list of the masked indexes inside the input_ids list
                """

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar("loss", loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update(batch_view_1, batch_view_2)  # update the key encoder
                niter += 1

            print("End of epoch {}".format(epoch_counter))

    def update(self, batch_view_1, batch_view_2):
        """
        change the prediction in all models 
        batch_view_1 include (online_network_inputs, online_network_labels)
        online_network = model(labels = online_network_labels['input_ids'])
        """
        online_network_inputs, online_network_labels, masks_o = batch_view_1
        target_network_inputs, target_network_labels, masks_t = batch_view_1

        # compute query feature
        predictions_from_view_1 = self.online_network(
            input_ids=online_network_inputs["input_ids"],
            attention_mask=online_network_inputs["attention_mask"],
            token_type_ids=online_network_inputs["token_type_ids"],
        )
        predictions_from_view_2 = self.online_network(batch_view_2)

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.boyl_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.boyl_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

        # save checkpoints
        # self.save_model(os.path.join(modeupdate)
