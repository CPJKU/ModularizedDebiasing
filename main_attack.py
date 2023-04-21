import os
import re
import ruamel.yaml as yaml
import argparse
from pathlib import Path
import torch

from src.training_logger import TrainLogger
from src.adv_attack import run_adv_attack
from src.model_functions import model_factory
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    set_optional_args
)

### DEFINE MANUALLY
CP_DIR = "/share/home/lukash/pan16/bertbase/cp_cp_init"
# CP = "task-baseline-bert_uncased_L-4_H-256_A-4-64-2e-05-seed4.pt"
CP = None #"task-diff_pruning_0.05-bert_uncased_L-4_H-256_A-4-64-2e-05-seed4.pt"
LOAD_CP_KWARGS = {} # {"remove_parametrizations": True}
### DEFINE MANUALLY

def main(checkpoint):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--seed", type=int, help="torch random seed")
    parser.add_argument("--ds", type=str, help="dataset")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--no_weighted_loss", action="store_true", help="do not use weighted loss for protected attribute")
    base_args, optional = parser.parse_known_args()

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{base_args.ds}"
    args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    set_optional_args(args_train, optional)

    if base_args.debug:
        set_num_epochs_debug(args_attack)
        set_dir_debug(args_attack)

    setattr(base_args, "adv", ("adverserial" in checkpoint))
    setattr(base_args, "modular", ("modular" in checkpoint))
    if base_args.seed is None:
        checkpoint_seed = int(re.search(r"(?<=seed)[\d+]", checkpoint).group())
        setattr(base_args, "seed", checkpoint_seed)

    if isinstance(args_train.protected_key, str):
        setattr(base_args, "prot_key_idx", None)
    else:
        keys_in_cp = []
        for k in args_train.protected_key:
            if k in checkpoint:
                keys_in_cp.append(k)
        if len(keys_in_cp)==1:
            idx = [i for i,x in enumerate(args_train.protected_key) if x==keys_in_cp[0]][0]
            setattr(base_args, "prot_key_idx", idx)
        else:
            setattr(base_args, "prot_key_idx", None)

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = get_device(not base_args.cpu, base_args.gpu_id)
    print(f"Device: {device}")

    trainer = model_factory(f"{CP_DIR}/{checkpoint}", **LOAD_CP_KWARGS)
    trainer.to(device)

    if hasattr(trainer, "adv_merged"):
        setattr(args_train, "modular_adv_merged", getattr(trainer, "adv_merged"))

    logger_name = "-".join([x for x in [
        f"only_adv_attack_{checkpoint[:-3]}",
        str(args_train.batch_size),
        str(args_attack.learning_rate),
        "weighted_loss_prot" if not base_args.no_weighted_loss else None,
        f"seed{base_args.seed}"
    ] if x is not None])
    train_logger = TrainLogger(
        log_dir = Path(args_train.log_dir),
        logger_name = logger_name,
        logging_step = args_attack.logging_step
    )

    print(f"running model {checkpoint}")
    print(base_args)
    print(args_train)
    print(args_attack)

    run_adv_attack(
        base_args,
        args_train,
        args_attack,
        trainer,
        train_logger
    )
    

if __name__ == "__main__":

    if CP is not None:
        main(CP)
    else:
        for checkpoint in os.listdir(CP_DIR):
            main(checkpoint)



