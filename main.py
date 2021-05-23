import pandas as pd
import torch
import wandb
from data import ConcatDataset, DoubleSynonymsDataset, preprocesser
from torch.nn import DataParallel
from transformers import (
    AlbertForMaskedLM,
    AlbertTokenizer,
    AlbertConfig,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AdamW,
)
import config_lm

from model import BoylLanguegeModel, MLPHead
from train import Trainer

wandb.init(project="byolm", name="BYOLM")

config = AutoConfig.from_pretrained(config_lm.model, output_hidden_states=True)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError("Must Use GPU")

print(f"Training with: {device}")

data_file = pd.read_csv(config_lm.data_path)

online_tokens, target_tokens = preprocesser(data_file)
online = DoubleSynonymsDataset(online_tokens)
target = DoubleSynonymsDataset(target_tokens)
dataset = ConcatDataset(online, target)

print("***** Done loading the data *****")

online_network = ByolLanguegeModel.from_pretrained("albert-large-v2", config=config)
online_network = torch.nn.DataParallel(online_network)

if load:
    try:
        checkpoints_folder = os.path.join(os.getcwd(), "checkpoints")

        load_params = torch.load(
            os.path.join(os.path.join(checkpoints_folder, "albert.model_2.pth")),
            map_location=torch.device(torch.device(device)),
        )

        online_network.load_state_dict(load_params["online_network_state_dict"])
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")
target_network = torch.nn.DataParallel(
    ByolLanguegeModel.from_pretrained("albert-large-v2", config=config)
)
predictor = torch.nn.DataParallel(
    MLPHead(
        in_channels=config.hidden_size,
        mlp_hidden_size=config.hidden_size * 12,
        projection_size=config.hidden_size,
    )
)
wandb.watch(online_network)

if config_lm.optimizer == "adam":
    optimizer = AdamW(
        list(online_network.parameters()) + list(predictor.parameters()),
        lr=config_lm.lr,
        weight_decay=config_lm.weight_decay,
    )

trainer = Trainer(
    online_network=online_network,
    target_network=target_network,
    optimizer=optimizer,
    device=device,
    m=config_lm.m,
    batch_size=config_lm.train_batch_size,
    num_workers=train_batch_size.num_workers,
    checkpoint_interval=config_lm.checkpoint_interval,
    max_epochs=train_batch_size.epochs,
    predictor=predictor,
)
trainer.train(dataset)
