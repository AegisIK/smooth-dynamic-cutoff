import os
import subprocess
import argparse
import signal
import sys

systems = {
    'Ac-Ala3-NHMe': 0.07,
    'DHA': 0.12,
    'stachyose': 0.29,
    'AT-AT': 0.15,
    'AT-AT-CG-CG': 0.20,
    'double-walled_nanotube': 0.16,
    'buckyball-catcher': 0.10
}


DATA_DIR = "[INSERT DATA DIR HERE]"
SAVE_DIR = "[INSERT CHECKPOINT DIR HERE]"

base_cmd = ["nequip-train", "-cp", "[INSERT PATH TO DYNAMIC CUTOFF REPO HERE]/training_nequip/", "-cn", "nequip_config_full.yaml"]

def run(add_on):
    print("\n===== RUNNING COMMAND =====")
    cmd = base_cmd + add_on
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

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=int, required=True)
args = parser.parse_args()

if args.config == 0:
    SELECTED_SYSTEMS = [0, 1]
    DEVICE = 0

elif args.config == 1:
    SELECTED_SYSTEMS = [2, 3]
    DEVICE = 1

elif args.config == 2:
    SELECTED_SYSTEMS = [4, 5, 6]
    DEVICE = 2

for i, system in enumerate(systems):
    if i not in SELECTED_SYSTEMS:
        continue
    print(f"---- Starting system {system} ----")
    
    for mode in ["dyn", "full"]:
        print(f"---- Starting mode {mode} ----")
        # Construct add on to base command
        add_ons = []
        add_ons += [f"data.split_dataset.file_path={os.path.join(DATA_DIR, f"md_22_{system}.xyz")}"]
        add_ons += [f"data.split_dataset.train={systems[system]}"]
        add_ons += [f"data.split_dataset.val={0.99 - systems[system]}"]
        add_ons += [f"data.split_dataset.test={0.01}"]

        add_ons += [f"trainer.devices=[{DEVICE}]"]
        add_ons += [f"trainer.logger.name=nequip_md22_{system}_full_rerun" if mode == "full" else f"trainer.logger.name=nequip_md22_{system}_dyn_20_rerun"]
        add_ons += [f"trainer.callbacks.0.dirpath={os.path.join(SAVE_DIR, system, mode)}"]

        add_ons += [f"training_module.optimizer.lr=0.01"]

        if mode == "dyn":
            add_ons += [f"training_module.model.dyn_cutoff=20"]
        else:
            add_ons += [f"training_module.model.dyn_cutoff=false"]

        run(add_ons)
