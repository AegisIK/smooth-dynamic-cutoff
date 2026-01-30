import argparse
import logging
import os
from typing import Dict, Optional, Tuple

import torch
import tqdm
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from orb_models.forcefield.pretrained import orb_v3_conservative_omol
from orb_models.forcefield.atomic_system import SystemConfig
from orb_models import utils
from orb_models.dataset import augmentations
from orb_models.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.forcefield import base, property_definitions


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    log_freq: int = 50,
) -> Dict[str, float]:
    """Run evaluation (validation) pass and return metrics.

    This runs with model.eval() and torch.no_grad().
    """

    model.eval()
    metrics = utils.ScalarMetricTracker()
    step = 0

    try:
        total = len(dataloader)
    except Exception:
        total = None

    for batch in tqdm.tqdm(dataloader, desc="Validation", total=total, leave=False):
        try:
            batch = batch.to(device)
            with torch.autocast("cuda", enabled=False):
                batch_outputs = model.loss(batch)
                metrics.update(batch_outputs.log)

        except Exception as e:
            # Reverting to your exact error handling logic
            raise e
            print(f"Got error during eval: {e}. Skipping and continuing regardless..")
        step += 1

    # final metrics for validation
    val_metrics = metrics.get_metrics()
    return val_metrics


def build_loaders(
    dataset_name: str,
    val_data_path: str,
    num_workers: int,
    batch_size: int,
    system_config: SystemConfig,
    target_config: Optional[Dict] = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Builds the train and validation dataloaders from separate paths."""
    
    
    # Instantiate target config
    target_config = property_definitions.instantiate_property_config(target_config)
    
    logging.info(f"Loading Validation Dataset from: {val_data_path}")
    val_dataset = AseSqliteDataset(
        dataset_name,
        val_data_path,
        system_config=system_config,
        target_config=target_config,
        **kwargs,
    )

    logging.info(f"Val size: {len(val_dataset)}")

    # Val loader: use SequentialSampler
    val_sampler = SequentialSampler(val_dataset)
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=max(1, min(4, num_workers)),
        worker_init_fn=utils.worker_init_fn,
        collate_fn=base.batch_graphs,
        timeout=0,
    )

    return val_loader


# -----------------------------------------------------------------------------
# MAIN EVALUATION LOGIC
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ORB model using training logic")
    
    # Args from your original script needed for setup
    parser.add_argument("--random_seed", default=1234, type=int)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--dataset", default="omol", type=str)
    
    # Data paths - pointing to the data you want to EVALUATE
    parser.add_argument("--data_path", required=True, type=str, help="Path to the .db file to evaluate.")
    
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    
    # Checkpoint to load
    parser.add_argument("--checkpoint_path", required=False, type=str, help="Path to model checkpoint.")
    parser.add_argument("--prune", action='store_true')

    args = parser.parse_args()

    print(f"Prune is {args.prune}")

    # 1. Init Device & Seed
    device = utils.init_device(device_id=args.device_id)
    utils.seed_everything(args.random_seed)

    # 2. Init Model (Same config as training)
    # Note: train=True is kept to match your training script instantiation exactly, 
    # though we call model.eval() later.
    model = orb_v3_conservative_omol(
        device=device,
        train=True, 
        prune=args.prune
    )
    model.to(device)

    # 3. Load Checkpoint
    if args.checkpoint_path:
        logging.info(f"Loading weights from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    # 4. Build Loaders 
    # We use your build_loaders function. We pass the same path for train/val 
    # because we only care about the 'val_loader' output for this script.
    graph_targets = ["energy", "stress"] if model.has_stress else ["energy"]
    
    loader_args = dict(
        dataset_name=args.dataset,
        val_data_path=args.data_path,   # This is what we evaluate
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        target_config={"graph": graph_targets, "node": ["forces"]},
    )

    logging.info("Building dataloaders...")
    val_loader = build_loaders(
        **loader_args,
        system_config=model.system_config,
    )

    # 5. Run Evaluation
    logging.info("Starting evaluation...")

    metrics = evaluate(model=model, dataloader=val_loader, device=device)

    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("="*40)

if __name__ == "__main__":
    main()