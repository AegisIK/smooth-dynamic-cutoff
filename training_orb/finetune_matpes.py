"""Finetuning loop with validation. (Updated: linear warmup + linear decay scheduler, num_epochs workflow)"""

import argparse
import logging
import os
from typing import Dict, Optional, Union, Tuple

import torch
import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Subset, SequentialSampler
from orb_models.forcefield.pretrained import orb_v3_conservative_architecture
from orb_models.forcefield.atomic_system import SystemConfig


try:
    import wandb
except ImportError:
    raise ImportError(
        "wandb is not installed. Please install it with `pip install wandb`."
    )
from wandb import wandb_run

from orb_models import utils
from orb_models.dataset import augmentations
from orb_models.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.forcefield import atomic_system, base, pretrained, property_definitions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

USE_DYN_CUT = None # Should be either weight_mean or None

def init_wandb_from_config() -> wandb_run.Run:
    """Initialise wandb."""
    wandb.init(
        dir=os.path.join(os.getcwd(), "orb_wandb"),
        name=f"orbv3_matpes_dyn" if USE_DYN_CUT else "orbv3_matpes_full",
        project="Max_Neighbors",
    )
    assert wandb.run is not None
    return wandb.run


def make_linear_warmup_decay_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    """
    Construct a LambdaLR scheduler that linearly warms up from 0 to 1 over `warmup_steps`,
    then linearly decays from 1 to 0 from warmup_steps -> total_steps.
    If warmup_steps == 0, this is a pure linear decay from step=0 -> total_steps.
    """

    def lr_lambda(current_step: int):
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # after warmup: linear decay to 0
        denom = max(1, total_steps - warmup_steps)
        decayed = float(max(0, total_steps - current_step)) / denom
        return decayed

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


def finetune(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    lr_scheduler: Optional[_LRScheduler] = None,
    num_steps_per_epoch: Optional[int] = None,
    clip_grad: Optional[float] = None,
    log_freq: float = 100,
    device: torch.device = torch.device("cpu"),
    epoch: int = 0,
):
    """Train for a single epoch using an explicit number of steps per epoch.

    If `num_steps_per_epoch` is None, it will try to use len(dataloader).
    """
    run: Optional[wandb_run.Run] = wandb.run

    if clip_grad is not None:
        hook_handles = utils.gradient_clipping(model, clip_grad)

    metrics = utils.ScalarMetricTracker()

    # Set the model to "train" mode.
    model.train()

    # Determine number of steps for this epoch
    if num_steps_per_epoch is not None:
        steps_in_epoch = num_steps_per_epoch
    else:
        try:
            steps_in_epoch = len(dataloader)
        except TypeError:
            raise ValueError("Dataloader has no length; pass `num_steps_per_epoch` to finetune.")

    batch_generator = iter(dataloader)
    batch_generator_tqdm = tqdm.tqdm(batch_generator, total=steps_in_epoch, desc=f"Train epoch {epoch}")

    i = 0
    batch_iterator = iter(batch_generator_tqdm)
    while i < steps_in_epoch:
        try:
            optimizer.zero_grad(set_to_none=True)

            step_metrics = {
                "batch_size": 0.0,
                "batch_num_edges": 0.0,
                "batch_num_nodes": 0.0,
            }

            # Reset metrics so that it reports raw values for each step but still do averages on
            # the gradient accumulation.
            if i % log_freq == 0:
                metrics.reset()

            batch = next(batch_iterator)
            batch = batch.to(device)
            step_metrics["batch_size"] += len(batch.n_node)
            step_metrics["batch_num_edges"] += batch.n_edge.sum()
            step_metrics["batch_num_nodes"] += batch.n_node.sum()

            with torch.autocast("cuda", enabled=False):
                batch_outputs = model.loss(batch)
                loss = batch_outputs.loss
                metrics.update(batch_outputs.log)
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            loss.backward()

            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metrics.update(step_metrics)

            if i != 0 and i % log_freq == 0:
                metrics_dict = metrics.get_metrics()
                if run is not None:
                    step_global = (epoch * steps_in_epoch) + i
                    if run.sweep_id is not None:
                        run.log(
                            {"loss": metrics_dict.get("loss", None)},
                            commit=False,
                        )
                    run.log(
                        {"step": step_global},
                        commit=False,
                    )
                    run.log(utils.prefix_keys(metrics_dict, "finetune_step"), commit=True)
        except Exception as e:
            print(f"Got error during training: {e}. Skipping step and continuing regardless...")

        # Finished a single full step!
        i += 1

    if clip_grad is not None:
        for h in hook_handles:
            h.remove()

    return metrics.get_metrics()


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    log_freq: int = 50,
) -> Dict[str, float]:
    """Run evaluation (validation) pass and return metrics.

    This runs with model.eval() and torch.no_grad().
    """
    run: Optional[wandb_run.Run] = wandb.run
    model.eval()
    metrics = utils.ScalarMetricTracker()
    step = 0

    try:
        total = len(dataloader)
    except Exception:
        total = None

    for batch in tqdm.tqdm(dataloader, desc="Validation", total=total):
        try:
            batch = batch.to(device)
            with torch.autocast("cuda", enabled=False):
                batch_outputs = model.loss(batch)
                # batch_outputs.log should be a dict of losses/metrics per batch
                metrics.update(batch_outputs.log)

            # optionally log intermediate validation progress to wandb (not committing)
            if run is not None and (step % log_freq == 0):
                # compute a partial aggregated metric dict and log it (commit=False)
                run.log({"validation/step_batch": step}, commit=False)
                partial = utils.prefix_keys(metrics.get_metrics(), "validation_partial")
                run.log(partial, commit=False)
        except Exception as e:
            print(f"Got error during eval: {e}. Skipping and continuing regardless..")
        step += 1

    # final metrics for validation
    val_metrics = metrics.get_metrics()
    # log them to wandb (commit=True)
    if run is not None:
        # We expect the training code to log a global 'step' metric elsewhere; don't override it here.
        run.log(utils.prefix_keys(val_metrics, "validation"), commit=True)

    model.train()  # restore training mode
    return val_metrics


def build_loaders(
    dataset_name: str,
    dataset_path: str,
    num_workers: int,
    batch_size: int,
    system_config: atomic_system.SystemConfig,
    augmentation: Optional[bool] = True,
    target_config: Optional[Dict] = None,
    val_fraction: float = 0.05,
    random_seed: int = 1234,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Builds the train and validation dataloaders from a config file.

    Validation is `val_fraction` of the full dataset (clamped to at least 1 sample).
    Returns (train_loader, val_loader).
    """
    log_train = "Loading datasets:\n"
    aug = []
    if augmentation:
        aug = [augmentations.rotate_randomly]

    target_config = property_definitions.instantiate_property_config(target_config)
    dataset = AseSqliteDataset(
        dataset_name,
        dataset_path,
        system_config=system_config,
        target_config=target_config,
        augmentations=aug,
        use_dynamic_cutoff=USE_DYN_CUT,
        **kwargs,
    )

    total_size = len(dataset)
    log_train += f"Total dataset size: {total_size} samples"
    logging.info(log_train)

    # Compute split sizes
    val_size = int(max(1, round(val_fraction * total_size)))
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError(
            f"Dataset too small for splitting: total_size={total_size}, val_fraction={val_fraction}"
        )

    # Deterministic split using a torch.Generator seeded with provided random seed
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    lengths = [train_size, val_size]
    train_subset, val_subset = torch.utils.data.random_split(dataset, lengths, generator=generator)

    logging.info(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}")

    # Train loader: use BatchSampler with RandomSampler on the train subset
    train_sampler = RandomSampler(train_subset)
    train_batch_sampler = BatchSampler(
        train_sampler,
        batch_size=batch_size,
        drop_last=False,
    )
    train_loader: DataLoader = DataLoader(
        train_subset,
        num_workers=num_workers,
        worker_init_fn=utils.worker_init_fn,
        collate_fn=base.batch_graphs,
        batch_sampler=train_batch_sampler,
        timeout=10 * 60 if num_workers > 0 else 0,
    )

    # Val loader: sequential sampling, simple DataLoader with same batch_size
    val_sampler = SequentialSampler(val_subset)
    val_loader: DataLoader = DataLoader(
        val_subset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=max(1, min(4, num_workers)),  # fewer workers for validation typically OK
        worker_init_fn=utils.worker_init_fn,
        collate_fn=base.batch_graphs,
        timeout=0,
    )

    return train_loader, val_loader



def run(args):
    """Training Loop.
    Uses args.num_epochs (number of epochs) and determines steps_per_epoch from train_loader length.
    If dataloader has no length, falls back to args.num_steps (legacy behavior).
    """
    device = utils.init_device(device_id=args.device_id)
    utils.seed_everything(args.random_seed)


    # Instantiate model
    model = orb_v3_conservative_architecture(
        latent_dim=128,
        base_mlp_hidden_dim=64,
        base_mlp_depth=2,
        head_mlp_hidden_dim=64,
        head_mlp_depth=2,
        num_message_passing_steps=5,
        activation="silu",
        has_charge_spin_cond=False,
        has_stress=False,
        device=device,
        system_config=SystemConfig(max_num_neighbors=1000, radius=6.0),
        loss_weights={"grad_forces": 10, "grad_stress": 0}
    )
    model.dynamic_cutoff = USE_DYN_CUT

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {model_params} trainable parameters.")

    # Move model to correct device.
    model.to(device=device)

    wandb_run = None
    # Logger instantiation/configuration
    if args.wandb:
        logging.info("Instantiating WandbLogger.")
        wandb_run = init_wandb_from_config()

        wandb.define_metric("step")
        wandb.define_metric("finetune_step/*", step_metric="step")
        # define validation metrics to use same step metric
        wandb.define_metric("validation/*", step_metric="step")
        wandb.define_metric("validation_partial/*", step_metric="step")

    graph_targets = ["energy", "stress"] if model.has_stress else ["energy"]
    loader_args = dict(
        dataset_name=args.dataset,
        dataset_path=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": graph_targets, "node": ["forces"]},
    )
    # Build both train and validation loaders
    train_loader, val_loader = build_loaders(
        **loader_args,
        system_config=model.system_config,
        augmentation=True,
        val_fraction=0.05,
        random_seed=args.random_seed,
    )
    logging.info("Starting training!")

    # Determine steps per epoch
    try:
        steps_per_epoch = len(train_loader)
    except Exception:
        if args.num_steps is None:
            raise ValueError("Cannot determine steps per epoch from dataloader and --num_steps not provided.")
        steps_per_epoch = args.num_steps
        logging.warning(f"Using legacy fallback steps_per_epoch={steps_per_epoch}")

    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    # Create optimizer and scheduler (we place scheduler creation after steps are known)
    optimizer, _ = utils.get_optim(args.lr, total_steps, model)  # ignore returned scheduler from utils.get_optim
    lr_scheduler = make_linear_warmup_decay_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    logging.info(f"Total steps: {total_steps}, steps_per_epoch: {steps_per_epoch}, warmup_steps: {warmup_steps}")

    for epoch in range(0, args.num_epochs):
        print(f"Start epoch: {epoch} training...")
        train_metrics = finetune(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            lr_scheduler=lr_scheduler,
            clip_grad=args.gradient_clip_val,
            device=device,
            num_steps_per_epoch=steps_per_epoch,
            epoch=epoch,
        )

        logging.info(f"Finished epoch {epoch} training. Train metrics: {train_metrics}")

        # Run validation after each epoch
        val_metrics = evaluate(model=model, dataloader=val_loader, device=device)
        logging.info(f"Validation metrics after epoch {epoch}: {val_metrics}")

        # Also log epoch-level metrics to wandb with a canonical 'step' if wandb is available
        if wandb_run is not None:
            # choose a step index for epoch-level logging (end of epoch)
            global_step = (epoch + 1) * steps_per_epoch
            wandb_run.log({"step": global_step}, commit=False)
            wandb_run.log(utils.prefix_keys(train_metrics, "train_epoch"), commit=False)
            wandb_run.log(utils.prefix_keys(val_metrics, "validation_epoch"), commit=True)

        # Save every `save_every_x_epochs` epochs and final epoch
        if (epoch % args.save_every_x_epochs == 0) or (epoch == args.num_epochs - 1):
            # create ckpts folder if it does not exist
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoint_path, f"checkpoint_epoch{epoch}.ckpt"),
            )
            logging.info(f"Checkpoint saved to {args.checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


def main():
    """Main."""
    parser = argparse.ArgumentParser(
        description="Finetune orb model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--random_seed", default=1234, type=int, help="Random seed for finetuning."
    )
    parser.add_argument(
        "--device_id", default=2 if USE_DYN_CUT else 3, type=int, help="GPU index to use if GPU is available." # YOU MAY NEED TO CHANGE THIS
    )
    parser.add_argument(
        "--wandb",
        default=True,
        action="store_true",
        help="If the run is logged to Weights and Biases (requires installation).",
    )
    parser.add_argument(
        "--dataset",
        default="r2scan-matpes",
        type=str,
        help="Dataset name for wandb run logging.",
    )
    parser.add_argument(
        "--data_path",
        default="[INSERT PATH TO DB HERE]",
        type=str,
        help="Dataset path to an ASE sqlite database (you must convert your data into this format).",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of cpu workers for the pytorch data loader.",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for finetuning."
    )
    parser.add_argument(
        "--gradient_clip_val", default=1, type=float, help="Gradient clip value."
    )
    parser.add_argument(
        "--num_epochs",
        default=5000,
        type=int,
        help="Number of epochs to finetune.",
    )
    parser.add_argument(
        "--save_every_x_epochs",
        default=500,
        type=int,
        help="Save model every x epochs.",
    )
    # legacy fallback: used only if dataloader has no __len__
    parser.add_argument(
        "--num_steps",
        default=0,
        type=int,
        help="(Legacy) Number of steps per epoch fallback if dataloader has no length.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=os.path.join("orb_checkpoints", "matpes_dyn" if USE_DYN_CUT else "matpes_full"),
        type=str,
        help="Path to save the model checkpoint.",
    )
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Learning rate. 3e-4 is purely a sensible default; you may want to tune this for your problem.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=500,
        type=float,
        help="Number of epochs to use for linear warmup (can be fractional)."
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
