import sys
sys.path.insert(0,'..')

import os
import argparse
import ruamel.yaml as yaml
import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_diff_modular import ModularDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from OLD_scripts.model_modular import ModularModel
from src.utils import (
    get_device,
    get_data,
    set_optional_args,
    dict_to_device
)


MODEL_PATH = "/share/home/lukash/{}/{}/cp"


@torch.no_grad()
def generate_embeddings(trainer_biased: nn.Module, trainer_debiased: nn.Module, loader: DataLoader):
    assert trainer_biased.device == trainer_debiased.device
    trainer_biased.eval()
    trainer_debiased.eval()
    emb_list_biased = []
    emb_list_debiased = []
    labels_task_list = []
    labels_protected_list = []
    for batch in tqdm(loader, desc="generating embeddings"):
        inputs, labels_task, labels_protected = batch
        inputs = dict_to_device(inputs, trainer_biased.device)
        emb_biased = trainer_biased._forward(**inputs)
        emb_debiased = trainer_debiased._forward(**inputs)

        emb_list_biased.append(emb_biased.cpu())
        emb_list_debiased.append(emb_debiased.cpu())
        labels_task_list.append(labels_task)
        labels_protected_list.append(labels_protected)
    return torch.cat(emb_list_biased), torch.cat(emb_list_debiased), torch.cat(labels_task_list), torch.cat(labels_protected_list)


def get_models(base_args):
    model_path = MODEL_PATH.format(base_args.ds, base_args.model_type)
    if base_args.modular:
        if base_args.baseline:
            model_name = f'bert_uncased_L-4_H-256_A-4-modular_baseline-seed{base_args.seed}.pt'
            model_biased = ModularModel.load_checkpoint(os.path.join(model_path, model_name), debiased=False)
            model_debiased = ModularModel.load_checkpoint(os.path.join(model_path, model_name), debiased=True)
        else:
            model_name = f'bert_uncased_L-4_H-256_A-4-modular_fixmask{base_args.fixmask_pct}-sparse_task-merged_head-seed{base_args.seed}.pt'
            model_biased = ModularDiffModel.load_checkpoint(os.path.join(model_path, model_name), remove_parametrizations=True, debiased=False)
            model_debiased = ModularDiffModel.load_checkpoint(os.path.join(model_path, model_name), remove_parametrizations=True, debiased=True)
    else:
        if base_args.baseline:
            model_name_biased = f'bert_uncased_L-4_H-256_A-4-task_baseline-seed{base_args.seed}.pt'
            model_name_debiased = f'bert_uncased_L-4_H-256_A-4-adv_baseline-seed{base_args.seed}.pt'
            model_biased = TaskModel.load_checkpoint(os.path.join(model_path, model_name_biased))
            model_debiased = AdvModel.load_checkpoint(os.path.join(model_path, model_name_debiased))
        else:
            model_name_biased = f'bert_uncased_L-4_H-256_A-4-task_fixmask{base_args.fixmask_pct}-seed{base_args.seed}.pt'
            model_name_debiased = f'bert_uncased_L-4_H-256_A-4-adv_fixmask{base_args.fixmask_pct}-seed{base_args.seed}.pt'
            model_biased = TaskDiffModel.load_checkpoint(os.path.join(model_path, model_name_biased), remove_parametrizations=True)
            model_debiased = AdvDiffModel.load_checkpoint(os.path.join(model_path, model_name_debiased), remove_parametrizations=True)
    return model_biased, model_debiased


def get_embeddings(base_args, args_train):
    emb_dir = os.path.join("..", "embeddings", base_args.ds, base_args.model_type)
    emb_type = f"modular{base_args.modular}_{'baseline' if base_args.baseline else 'fixmask'}{'' if base_args.baseline else base_args.fixmask_pct}_seed{base_args.seed}"
    try:
        train_embeddings_ds = torch.load(os.path.join(emb_dir, f"train_embeddings_ds_{emb_type}.pth"))
        val_embeddings_ds = torch.load(os.path.join(emb_dir, f"val_embeddings_ds_{emb_type}.pth"))
    except FileNotFoundError:
        torch.manual_seed(base_args.seed)
        print(f"torch.manual_seed({base_args.seed})")
        device = get_device(not base_args.cpu, base_args.gpu_id)
        print(f"Device: {device}")

        train_loader, val_loader, _, _ = get_data(args_train, debug=False)
        model_biased, model_debiased = get_models(base_args)
        model_biased.to(device)
        model_debiased.to(device)
        train_embeddings_biased, train_embeddings_debiased, train_labels_task, train_labels_protected = \
            generate_embeddings(model_biased, model_debiased, train_loader)
        val_embeddings_biased, val_embeddings_debiased, val_labels_task, val_labels_protected = \
            generate_embeddings(model_biased, model_debiased, val_loader)
        del model_biased
        del model_debiased

        train_embeddings_ds = TensorDataset(train_embeddings_biased, train_embeddings_debiased, train_labels_task, train_labels_protected)
        val_embeddings_ds = TensorDataset(val_embeddings_biased, val_embeddings_debiased, val_labels_task, val_labels_protected)

        os.makedirs(emb_dir, exist_ok=True)
        torch.save(train_embeddings_ds, os.path.join(emb_dir, f"train_embeddings_ds_{emb_type}.pth"))
        torch.save(val_embeddings_ds, os.path.join(emb_dir, f"val_embeddings_ds_{emb_type}.pth"))

    return train_embeddings_ds, val_embeddings_ds


def main(base_args):

    with open(os.path.join("..", "cfg.yml"), "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{base_args.ds}"
    args_train = argparse.Namespace(**cfg[data_cfg], **cfg["model_config"])

    return get_embeddings(base_args, args_train)


def main_wrapper(args):
    combs = [
        {"fixmask_pct": 0.1, "baseline": False, "modular": False},
        {"fixmask_pct": 0.05, "baseline": False, "modular": False},
        {"baseline": True, "modular": False},
        {"fixmask_pct": 0.1, "baseline": False, "modular": True},
        {"fixmask_pct": 0.05, "baseline": False, "modular": True},
        {"baseline": True, "modular": True}
    ]
    seeds = list(range(5))
    datasets = ["bios", "pan16"]

    combs = [{**comb[0], "seed": comb[1], "ds": comb[2]} for comb in itertools.product(combs, seeds, datasets)]

    for comb in combs:
        comb["model_type"] = args.model_type
        comb["cpu"] = args.cpu
        comb["gpu_id"] = args.gpu_id

        args_ = argparse.Namespace(**comb)
        main(args_)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bertl4", help="bertbase or bertl4")
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--run_all", action="store_true", help="runs all experiments for one model type")
    parser.add_argument("--fixmask_pct", type=float, default=0.1, help="for diff models, which sparsity percentage")
    parser.add_argument("--baseline", action="store_true", help="Set to True if you want to run baseline models (no diff-pruning)")
    parser.add_argument("--modular", action="store_true", help="Whether to run modular training (task only and adverserial)")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")

    args = parser.parse_args()

    if args.run_all:
        main_wrapper(args)
    else:
        main(args)