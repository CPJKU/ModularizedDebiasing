# Diff-Pruning with Adverserial Training

This code implements various models related to adverserial training and diff-pruning (Guo et. al., 2020)

## Installation

To run the code make sure conda is installed and then run

```bash
conda env create -f environment.yml
```

Then activate the environment by running

```bash
conda activate diff_pruning
```

## Architecture

The project structure looks as follows

📦debiasing_text_classification \
 ┣ 📂src \
 ┃ ┣ 📂models (directory which contains all model classes)\
 ┃ ┃ ┣ 📜model_adv.py (baseline model for adverserial training) \
 ┃ ┃ ┣ 📜model_base.py (contains base classes with methods that are used by all models) \
 ┃ ┃ ┣ 📜model_diff_adv.py (model with 2 subnetworks for adverserial training) \
 ┃ ┃ ┣ 📜model_diff_task.py (model with subnetwork for task training) \
 ┃ ┃ ┣ 📜model_heads.py (classifier and adverserial head classes) \
 ┃ ┃ ┣ 📜model_task.py (baseline model for task training) \
 ┃ ┃ ┗ 📜weight_parametrizations.py (contains weight parametrizations for subnetwork training*) \
 ┃ ┣ 📜adv_attack.py (contains function to run adverserial attack) \
 ┃ ┣ 📜data_handler.py \
 ┃ ┣ 📜metrics.py \
 ┃ ┣ 📜training_logger.py \
 ┃ ┗ 📜utils.py \
 ┣ 📜AdvBERT_script.py (testing script for training to verify modular code is working)\
 ┣ 📜AdvBERT_script2.py (second testing script for training which contains old model classes from main branch)\
 ┣ 📜cfg.yml (hyperparameters)\
 ┣ 📜environment.yml (conda environment config)\
 ┣ 📜main.py (main file to run experiments with)\
 ┣ 📜main_attack.py (used to run an adverserial attack only using a model checkpoint)\
 ┗ 📜readme.md

\* Weight parametrizations are implemented as modules and use pytorch parametrizations functionality [LINK](https://pytorch.org/tutorials/intermediate/parametrizations.html)

## cfg.yml

contains hyperparameter configuration

* data_config \
filepaths to data files
* model_config \
name of pretrained model and batch_size to use
* train_config_diff_pruning \
hyperparameters for diff-pruning-models (model_diff_adv.py and model_diff_task.py)
* train_config_baseline \
hyperparameters for baseline models (model_adv.py and model_task.py)
* adv_attack
hyperparameters for adverserial attack

## Usage

```bash
python3 main.py
```

Optional arguments with example inputs

* --baseline=True \
Run baseline or diff-pruning
* --adv=True \
Run adverserial training
* --gpu_id 0 1 2 3 \
Which gpus to run experiment on (can be multiple)
* --debug=True \
To verify code can run through, limits number of batches which are used to 10
* --run_adv_attack=False \
Set to false if you do not want to run adverserial attack after training