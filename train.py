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


class Trainer:
    """
    main training class
    """

    def __init__(
        self,
        online_network,
        target_network,
        predictor,
        optimizer,
        device,
        max_epochs,
        m,
        batch_size,
        num_workers,
        checkpoint_interval,
        eval_during_training=False,
    ):

        self.optimizer = optimizer
        self.device = device
        self.online_network = online_network.to(self.device)
        self.target_network = target_network.to(self.device)
        self.predictor = predictor.to(self.device)
        self.writer = SummaryWriter()
        self.max_epochs = (max_epochs,)
        self.m = (m,)
        self.batch_size = (batch_size,)
        self.num_workers = (num_workers,)
        self.checkpoint_interval = checkpoint_interval
        self.eval_during_training = eval_during_training

    def update_moving_average(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data = param_k.data * 0.996 + param_q.data * (1.0 - 0.996)

    def boyl_loss(self, x, y):
        if x.shape[0] != y.shape[0]:
            l = min(x.shape[0], y.shape[0])
            x = x.narrow(0, 0, l)
            y = y.narrow(0, 0, l)
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, dataset):
        # change the params to config
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [int(len(dataset) * 0.95), len(dataset) - int(len(dataset) * 0.95)]
        )
        train_dataset = dataset
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=4, drop_last=False,
        )
        test_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=4, drop_last=False,
        )
        model_checkpoints_folder = os.path.join(os.getcwd(), "checkpoints")

        self.initializes_target_network()
        # cahnge to config
        for niter, epoch_counter in enumerate(range(7)):
            epoch_training_loss = 0
            epoch_validation_loss = 0
            self.online_network.train()
            self.target_network.train()
            self.predictor.train()
            print(f"Epoch: {epoch_counter}")
            start = time.time()
            for iter_train, batch in enumerate(train_loader):
                """
                batch:
                - Online batch
                    zipped together
                    - ids
                    - masks
                """

                loss = self.update_train(batch)
                epoch_training_loss += loss
                self.writer.add_scalar("loss", loss, global_step=niter)
                self.optimizer.zero_grad()
                loss.backward()

                self.update_moving_average()
                self.optimizer.step()  # update the key encoder
                wandb.log({"iteration_batch": iter_train, "training_loss_batch": loss})

            if self.eval_during_training:
                self.online_network.eval()
                self.target_network.eval()
                self.predictor.eval()
                for iter_val, batch_val in enumerate(test_loader):
                    loss_val = self.update_val(batch_val)
                    epoch_validation_loss += loss_val
                    wandb.log(
                        {"iteration_batch": iter_val, "validation_loss_batch": loss_val}
                    )

                epoch_validation_loss /= batch_size
            epoch_training_loss /= batch_size
            wandb.log({"Epoch": epoch_counter, "training_loss": epoch_training_loss})
            print(
                f"**** current training loss for epoch {epoch_counter} is: {epoch_training_loss} ****"
            )

            if self.eval_during_training:
                wandb.log(
                    {"iteration": niter, "validation_loss": epoch_validation_loss}
                )
                print(
                    f"**** current val loss for epoch {epoch_counter} is: {epoch_validation_loss} ****"
                )

            print("End of epoch {}".format(epoch_counter))
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print(
                "This epoch took: {:0>2}:{:0>2}:{:05.2f}".format(
                    int(hours), int(minutes), seconds
                )
            )
            self.save_model(
                os.path.join(
                    model_checkpoints_folder, f"albert.model_{epoch_counter}.pth"
                )
            )
            self.online_network.module.save_pretrained(
                os.path.join(model_checkpoints_folder, f"albert_{epoch_counter}.bin")
            )

        print(f"using learning rate: {lr}. and {optim} optimizer")

    def update_train(self, batch):
        """
        change the prediction in all models
        batch_view_1 include (online_network_inputs, online_network_labels)
        online_network = model(labels = online_network_labels['input_ids'])
        """

        online, target = batch
        online_ids, online_masks = online
        target_ids, target_masks = target

        input_ids_view_1 = online_ids.to(device)
        input_ids_view_2 = target_ids.to(device)
        masked_indexes_view_1 = online_masks.to(device)
        masked_indexes_view_2 = target_masks.to(device)
        # compute query feature
        predictions_from_view_1 = self.predictor(
            self.online_network(
                input_ids=input_ids_view_1,
                masked_index=masked_indexes_view_1,
                output_hidden_states=True,
                output_attentions=True,
            )[1]
        )

        predictions_from_view_2 = self.predictor(
            self.online_network(
                input_ids=input_ids_view_2,
                masked_index=masked_indexes_view_2,
                output_hidden_states=True,
                output_attentions=True,
            )[1]
        )

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(
                input_ids=input_ids_view_1,
                masked_index=masked_indexes_view_1,
                output_hidden_states=True,
                output_attentions=True,
            )[1]

            targets_to_view_1 = self.target_network(
                input_ids=input_ids_view_2,
                masked_index=masked_indexes_view_2,
                output_hidden_states=True,
                output_attentions=True,
            )[1]

        loss = self.boyl_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.boyl_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def update_val(self, batch):
        """
        change the prediction in all models
        batch_view_1 include (online_network_inputs, online_network_labels)
        online_network = model(labels = online_network_labels['input_ids'])
        """

        online, target = batch
        online_ids, online_masks = online
        target_ids, target_masks = target

        input_ids_view_1 = online_ids.to(device)
        input_ids_view_2 = target_ids.to(device)
        masked_indexes_view_1 = online_masks.to(device)
        masked_indexes_view_2 = target_masks.to(device)
        # compute query feature

        # compute key features
        with torch.no_grad():
            predictions_from_view_1 = self.predictor(
                self.online_network(
                    input_ids=input_ids_view_1,
                    masked_index=masked_indexes_view_1,
                    output_hidden_states=True,
                    output_attentions=True,
                )[1]
            )

            predictions_from_view_2 = self.predictor(
                self.online_network(
                    input_ids=input_ids_view_2,
                    masked_index=masked_indexes_view_2,
                    output_hidden_states=True,
                    output_attentions=True,
                )[1]
            )

            targets_to_view_2 = self.target_network(
                input_ids=input_ids_view_1,
                masked_index=masked_indexes_view_1,
                output_hidden_states=True,
                output_attentions=True,
            )[1]

            targets_to_view_1 = self.target_network(
                input_ids=input_ids_view_2,
                masked_index=masked_indexes_view_2,
                output_hidden_states=True,
                output_attentions=True,
            )[1]

        loss = self.boyl_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.boyl_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save(
            {
                "online_network_state_dict": self.online_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            PATH,
        )
