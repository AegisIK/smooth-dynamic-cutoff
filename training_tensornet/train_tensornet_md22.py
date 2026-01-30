import os
os.environ["DGL_BACKEND"] = "pytorch"
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

from openqdc.datasets import MD22
# To suppress warnings for clearer output
warnings.simplefilter("ignore")
import torch
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=int)
args = parser.parse_args()

torch.set_float32_matmul_precision('highest')

if int(args.config) == 0:
    SELECTED_SYSTEMS = [0, 1]
    DEVICE = 0

elif int(args.config) == 1:
    SELECTED_SYSTEMS = [2, 3]
    DEVICE = 1
elif int(args.config) == 2:
    SELECTED_SYSTEMS = [4, 5, 6]
    DEVICE = 2
else:
    raise Exception(f"Weird config: {args.config}")

os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE) # you can set this to include/exclude gpus

systems = {'Ac-Ala3-NHMe': 0.07, 'DHA': 0.12, 'stachyose': 0.29, 'AT-AT': 0.15,
            'AT-AT-CG-CG': 0.20,
            'double-walled_nanotube': 0.16,
            'buckyball-catcher': 0.10}



for i, system in enumerate(systems.keys()):
    if not i in SELECTED_SYSTEMS:
        continue
    print(f"---------------- STARTING SYSTEM {system} ----------------")

    for mode in ["full", "dyn"]:
        print(f" ------ {mode} mode -----")
        # Load the dataset
        element_types = DEFAULT_ELEMENTS
        print("Loading dataset...")
        dataset = MGLDataset(
            include_line_graph=False,
            save_dir="[INSERT PATH TO DATA DIR HERE]",
            directory_name=f'md22_{system}',
            save_cache=True
        )

        training_set, validation_set, test_set = split_dataset(
            dataset, 
            frac_list=[systems[system], 0.01, 1 - 0.01 - systems[system]], 
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
            batch_size=2,
            num_workers=0,
        )

        forces1 = torch.concatenate(forces)
        rms_forces = torch.sqrt(torch.sum(forces1[:, 0] ** 2.0 + forces1[:, 1] ** 2.0 + forces1[:, 2] ** 2.0) / forces1.shape[0])
        del forces

        if mode == "full":
            model = TensorNet(max_neighbors=False, cutoff=6, units=64, nblocks=6)
            model_name = f"{system}_full"
        else:
            model = TensorNet(max_neighbors=True, cutoff=6, units=64, nblocks=6, weight_mean=20)
            model_name = f"{system}_mean_20"

        lit_model = PotentialLightningModule(
            model=model,
            element_refs = element_refs.property_offset,
            data_std = rms_forces,
            lr=1e-3,
            decay_steps=300,
            loss="huber_loss",
            loss_params={'delta':0.5},
            include_line_graph=False,
            force_weight=1000
        )

        logger = WandbLogger(name=f"tensornet_6_64_6_456k_{model_name}", project="Max_Neighbors", log_model="final", reinit=True)

        checkpoint = ModelCheckpoint(
            monitor="val_Total_Loss",
            every_n_epochs=1,
            save_top_k=1,
            save_last=True,
            mode="min",
            dirpath="checkpoints/" + model_name,
            filename="{epoch:04d}-best_model-tensornet-" + model_name,
        )

        trainer = pl.Trainer(
            max_epochs=2000,
            accelerator="cuda",
            logger=logger,
            callbacks=[EarlyStopping(monitor="val_Total_Loss", mode="min", patience=30), checkpoint],
            devices=[0],
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

        wandb.finish()
