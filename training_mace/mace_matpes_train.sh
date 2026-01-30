#!/bin/sh

CONFIG="$1"

# check if the argument was provided
if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <value>"
    exit 1
fi

case "$CONFIG" in
    0)
        echo "Training dynamic cutoff on MATPES"
        export CUDA_VISIBLE_DEVICES="0"

        python [INSERT PATH TO MACE REPO HERE]/run_train.py --E0s=average --model=ScaleShiftMACE --num_interactions=2 --num_channels=64 --max_L=0 --correlation=2 --r_max=6.0 --forces_weight=1000 --energy_weight=10 --batch_size=16 --valid_batch_size=16 --max_num_epochs=2000 --start_swa=450 --scheduler_patience=5 --patience=15 --eval_interval=3 --ema --swa --swa_forces_weight=10 --error_table=PerAtomMAE --default_dtype=float32 --seed=123 --wandb --wandb_project=Max_Neighbors2 --wandb_entity=[INSERT WANDB ACCOUNT HERE] --energy_key=energy --forces_key=forces --num_workers=0 --device=cuda --restart_latest --name="mace_matpes_dyn_20" --train_file=[INSERT TRAIN FILE PATH HERE] --valid_fraction=0.001 --wandb_name=mace_matpes_dyn_40 --checkpoints_dir=[INSERT CHECKPOINTS PATH HERE] --enable_cueq=True --prune --weight_mean 40

        ;;
    1)
        echo "Training full conv on MATPES"
        export CUDA_VISIBLE_DEVICES="1"

        python [INSERT PATH TO MACE REPO HERE]/run_train.py --E0s=average --model=ScaleShiftMACE --num_interactions=2 --num_channels=64 --max_L=0 --correlation=2 --r_max=6.0 --forces_weight=1000 --energy_weight=10 --batch_size=16 --valid_batch_size=16 --max_num_epochs=2000 --start_swa=450 --scheduler_patience=5 --patience=15 --eval_interval=3 --ema --swa --swa_forces_weight=10 --error_table=PerAtomMAE --default_dtype=float32 --seed=123 --wandb --wandb_project=Max_Neighbors2 --wandb_entity=[INSERT WANDB ACCOUNT HERE] --energy_key=energy --forces_key=forces --num_workers=0 --device=cuda --restart_latest --name="mace_matpes_full" --train_file=[INSERT TRAIN FILE PATH HERE] --valid_fraction=0.001 --wandb_name=mace_matpes_dyn_40 --checkpoints_dir=[INSERT CHECKPOINTS PATH HERE] --enable_cueq=True

        ;;
    *)
        echo "Invalid mode: $CONFIG (must be 0 or 1)"
        exit 1
        ;;
esac
