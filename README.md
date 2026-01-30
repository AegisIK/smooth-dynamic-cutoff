# Smooth Dynamic Cutoffs for Machine Learning Interatomic Potentials

Open source repository for the paper titled: ``Smooth Dynamic Cutoffs for Machine Learning Interatomic Potentials``

Authored by: Kevin Han, Haolin Cong, Bowen Deng, Amir Barati Farimani

Arxiv: https://www.arxiv.org/pdf/2601.21147
<p align="center">
      <img width="505.65" height="333.333" alt="image" src="https://github.com/user-attachments/assets/f57aa8e7-fb98-449c-bb40-921df537b7e9">

## Repo Layout
The repository contains 4 folders (``mace (MACE)``, ``nequip (Nequip)``, ``orb-models (Orbv3)``, ``matgl (TensorNet)``) corresponding to each of the 4 models trained in the paper. Each repository contains the modified dynamic cutoff (the function is called `fast_prune`) that is enabled typically through an argument in the constructor of each of the models.

The repository also contains 4 folders titled `training_mace`, `training_nequip`, `training_orb`, and `training_tensornet` which correspond to training code for each of the models.

## Installation and Setup
Due to dependency conflicts between repositories of the 4 models, we recommend creating a dedicated python environment for each model and running `pip install -e .` on only the model repository that you're interested in. We use python==3.12 for all environments and the default settings for each of the repositories.

⚠️⚠️ **We highly recommend you take a look at the structure of the training code and modify it to fit your needs.** ⚠️⚠️

We also recommend evaluating all models using the provided ASE Calculator interface.

## MACE
* Training scripts for both the MatPES and MD22 configurations are found in `training_mace`. 
* You will have to modify the scripts with your specific file paths. The code that has to be changed is labeled with `[INSERT ___ HERE]`. 
* Information on creating the datasets for ingestion are found in the MACE docs: https://mace-docs.readthedocs.io/en/latest/guide/training.html.

* Note that we did modify the MACE training script as well as the graph construction and architecture in orde to input the dynamic cutoff parameters into the training pipeline. For the MatPES dataset, we recommend evaluating the MACE model via ASE. For the MD22 dataset, the training logs can be used to read off validation results.

## Nequip
* Training scripts are found in `training_nequip`. 
* You will also have to run `pip install -e .` on the `nequip` folder in this repository before training as well. 
* The MatPES configuration are found in `nequip_config_dyn.yaml` and `nequip_config_full.yaml` and can be run via `nequip-train -cp full/path/to/config/directory -cn nequip_config_full.yaml`. More details can be found in the Nequip docs: https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html. Code that has to be changed is labeled with `[INSERT ___ HERE]`. 
* The MD22 training runs are performed by programmatically modifying and running the config files via `nequip_train_md22.py`.

## Orbv3
* Run `pip install -e .` on the `orb-models` folder.
* The orb-models repository doesn't supply code for pretraining. As a result, we modified the finetuning script to support pretraining. However, we do not perform their diffusion-based pretraining described in the Orbv1 paper due to a lack of an open source implementation.
* Orbv3 expects an ASE SQLite database for training data. More details on how to construct this database are here: https://github.com/orbital-materials/orb-models/blob/main/FINETUNING_GUIDE.md.
* The code that has to be changed is labeled with `[INSERT __ HERE]`.
* MD22 training is found in `finetune_md22.py`. MatPES training is found in `finetune_matpes.py`. Note that MatPES training will require you to enter your hyperparameters via the command line. 

## TensorNet (MatGL)
* We use the DGL version of TensorNet. Note that DGL is no longer being supported and does not support later versions of PyTorch.
* Install DGL and PyTorch: https://www.dgl.ai/pages/start.html
* Run `pip install -e .` on the `matgl` folder.
* Training requires creating an MGLDataset. Details on creating this dataset are found here: https://matgl.ai/tutorials%2FTraining%20a%20M3GNet%20Potential%20with%20PyTorch%20Lightning.html
* Training on the MatPES dataset are found in the `train_tensornet.py` file. Training with MD22 are found in `train_tensornet_md22.py`.

## Citing Us
If you find our work interesting and useful, please cite us here: 
```
@misc{han2026smoothdynamiccutoffsmachine,
      title={Smooth Dynamic Cutoffs for Machine Learning Interatomic Potentials}, 
      author={Kevin Han and Haolin Cong and Bowen Deng and Amir Barati Farimani},
      year={2026},
      eprint={2601.21147},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.21147}, 
}
```

## Questions/Contact Us
If you have any questions, feel free to either raise a GitHub issue on the repoisitory or email `kevinhan@cmu.edu`. The development of this work took place over 4 different servers so there may be discrepancies with versioning. If there is some discrepancy, please do let us know ASAP. Emailing directly will most likely get the fastest response! 
