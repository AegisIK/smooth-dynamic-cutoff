import os
import subprocess
import argparse
import signal
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=int)
args = parser.parse_args()

base_cmd = ["conda", "run", "-n", "[INSERT CONDA ENVIRONMENT NAME HERE]", "python", "[INSERT PATH TO MACE REPOSITORY HERE]run_train.py",
    "--E0s=average",
    "--model=MACE",
    "--num_interactions=2",
    "--num_channels=128",
    "--max_L=2",
    "--correlation=2",
    "--r_max=6.0",
    "--forces_weight=1000",
    "--energy_weight=10",
    "--batch_size=32",
    "--valid_batch_size=2",
    "--max_num_epochs=500",
    "--start_swa=450",
    "--scheduler_patience=5",
    "--patience=15",
    "--eval_interval=3",
    "--ema",
    "--swa",
    "--swa_forces_weight=10",
    "--error_table=PerAtomMAE",
    "--default_dtype=float32",
    "--seed=123",
    "--restart_latest",
    "--save_cpu",
    "--wandb",
    "--wandb_project=[INSERT WANDB PROJECT NAME HERE]",
    "--wandb_entity=[INSERT WANDB ACCOUNT NAME HERE]",
    "--energy_key=energy",
    "--forces_key=forces"
]

systems = {
    'Ac-Ala3-NHMe': 0.07,
    'DHA': 0.12,
    'stachyose': 0.29,
    'AT-AT': 0.15,
    'AT-AT-CG-CG': 0.20,
    'double-walled_nanotube': 0.16,
    'buckyball-catcher': 0.10
}


DATA_DIR = "[INSERT PATH TO DATA DIRECTORY HERE]"

def run(cmd):
    print("\n===== RUNNING COMMAND =====")
    print(" ".join(cmd))
    print("===========================\n")

    p = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        preexec_fn=os.setsid
    )

    try:
        p.wait()
    except KeyboardInterrupt:
        print("\nCTRL-C detected! Killing subprocess...")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        raise

if args.config == 0:
    SELECTED_SYSTEMS = [0]
    DEVICE = 0 # MAY NEED TO BE CHANGED 

elif args.config == 1:
    SELECTED_SYSTEMS = [1, 2]
    DEVICE = 1 # MAY NEED TO BE CHANGED 

elif args.config == 2:
    SELECTED_SYSTEMS = [3, 4]
    DEVICE = 2 # MAY NEED TO BE CHANGED 

elif args.config == 3:
    SELECTED_SYSTEMS = [5, 6]
    DEVICE = 3 # MAY NEED TO BE CHANGED 


for i, system in enumerate(systems):
    if i not in SELECTED_SYSTEMS:
        continue

    train_frac = systems[system]
    xyz_path = f"{DATA_DIR}/md_22_{system}.xyz"

    print(f" ---------- Starting system {system} ---------- ")
    for conv_type in ["dyn", "full"]:
        print(f" ---------- Starting mode {conv_type} ----------")
        run_name = f"mace_md22_{system}_{"full" if conv_type == "full" else "dyn_20"}"
        cmd = base_cmd.copy()

        cmd.append(f"--name={run_name}")
        cmd.append(f"--train_file={xyz_path}")
        cmd.append(f"--valid_fraction={1 - train_frac}")
        cmd.append(f"--wandb_name={run_name}")
        cmd.append(f"--checkpoints_dir=[INSERT CHECKPOINT DIRECTORY HERE]/{system}/{conv_type}")
        
        # Insert device id
        cmd.insert(4, f"CUDA_VISIBLE_DEVICES={DEVICE}")

        # Add pruned flags if needed
        if conv_type == "dyn":
            cmd.extend(["--prune", "--weight_mean", "20"])

        # Run the job
        run(cmd)
