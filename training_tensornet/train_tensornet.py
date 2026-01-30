import os
os.environ["DGL_BACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = "7" # you can set this to include/exclude gpus
import datetime
import warnings
import json
import matgl

from tqdm import trange
from functools import partial
import numpy as np
import torch
import lightning as pl
from dgl.data.utils import Subset
from dgl.data.utils import split_dataset
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from matgl.layers import AtomRef
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.utils.training import PotentialLightningModule, xavier_init
from torch.optim.lr_scheduler import CosineAnnealingLR
from matgl.config import DEFAULT_ELEMENTS
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from dataclasses import dataclass, asdict
import wandb
from matgl.models import CHGNet, TensorNet

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

import torch

torch.set_float32_matmul_precision('highest')

element_types = DEFAULT_ELEMENTS
print("Loading dataset...")
dataset = MGLDataset(
    include_line_graph=False,
    save_dir='[PATH TO DATA DIR]',
    directory_name='matpes_r2scan_6_train',
    save_cache=True
)

from functools import partial
training_set, validation_set, test_set = split_dataset(
    dataset, 
    frac_list=[0.99, 0.005, 0.005], 
    random_state=42, 
    shuffle=True
)

train_graphs = []
energies = []
forces = []
for (g, lat, attrs, lbs) in training_set:
    train_graphs.append(g)
    energies.append(lbs["energies"])
    forces.append(lbs['forces'])

element_refs = AtomRef(torch.zeros(89))
element_refs.fit(train_graphs, torch.hstack(energies))
collate_fn = partial(collate_fn_pes, include_line_graph=False, include_stress=False, include_magmom=False)
train_loader, val_loader = MGLDataLoader(
    train_data=training_set,
    val_data=validation_set,
    collate_fn=collate_fn,
    batch_size=128,
    num_workers=0,
)

forces1 = torch.concatenate(forces)
rms_forces = torch.sqrt(torch.sum(forces1[:, 0] ** 2.0 + forces1[:, 1] ** 2.0 + forces1[:, 2] ** 2.0) / forces1.shape[0])
del forces


model = TensorNet(max_neighbors=True, cutoff=6, units=64, nblocks=6, weight_mean=40)

lit_model = PotentialLightningModule(
    model=model,
    element_refs = element_refs.property_offset,
    data_std = rms_forces,
    lr=1e-3,
    decay_steps=1000,
    loss="huber_loss",
    loss_params={'delta':0.5},
    include_line_graph=False,
)

# set up your wandb login settings so that this works
logger = WandbLogger(name="[INSERT NAME OF WANDB RUN]", project="Max_Neighbors", log_model="final")

checkpoint = ModelCheckpoint(
    monitor="val_Total_Loss",
    every_n_epochs=1,
    save_top_k=1,
    save_last=True,
    mode="min",
    filename="{epoch:04d}-best_model",
)

# multi-gpu strategy
strategy = DDPStrategy(
    find_unused_parameters=True,
    timeout=datetime.timedelta(seconds=3 * 1800)
)

trainer = pl.Trainer(
    max_epochs=2000,
    accelerator="cuda",
    logger=logger,
    callbacks=[EarlyStopping(monitor="val_Total_Loss", mode="min", patience=30), checkpoint],
    devices=2,
    strategy=strategy,
    gradient_clip_val=2.0,
    accumulate_grad_batches=1,
    profiler="simple"
)

# this is to record gradients
logger.watch(lit_model)

# Staring from checkpoint
trainer.fit(
    model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)
