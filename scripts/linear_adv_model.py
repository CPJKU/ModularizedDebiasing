import sys
sys.path.insert(0,'..')

import os
import argparse
import math
import ruamel.yaml as yaml
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from src.adv_attack import get_hidden_dataloader
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_diff_modular import ModularDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from OLD_scripts.model_modular import ModularModel
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    get_data,
    get_callables,
    set_optional_args,
    dict_to_device,
    get_num_labels
)


EMB_DIR = "embeddings/bios"
EMB_TYPE = "modularFalse_baseline"


class LinearAdvModel(nn.Module):
    def __init__(self, emb_size: int, task_head: nn.Module, adv_head: nn.Module):
        super().__init__()
        self.linear_map = nn.Linear(emb_size, emb_size)
        self.task_head = task_head
        self.adv_head = adv_head

    def forward(x):
        emb_adv = self.linear_map(x)
        res_task = self.task_head(emb_adv)
        res_adv = self.adv_head(emb_adv)
        return emb_adv, res_task, res_adv


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bertbase", action="store_true", help="bert-base instead of bert-L4")
    parser.add_argument("--fixmask_pct", type=float, default=0.1, help="for diff models, which sparsity percentage")
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--baseline", action="store_true", help="Set to True if you want to run baseline models (no diff-pruning)")
    parser.add_argument("--modular", action="store_true", help="Whether to run modular training (task only and adverserial)")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    base_args, optional = parser.parse_known_args()

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{base_args.ds}"
    args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    set_optional_args(args_train, optional)

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = get_device(not base_args.cpu, base_args.gpu_id)
    print(f"Device: {device}")

    train_embeddings_ds = torch.load(os.path.join(EMB_DIR, f"train_embeddings_ds_{EMB_TYPE}.pth"))
    val_embeddings_ds = torch.load(os.path.join(EMB_DIR, f"val_embeddings_ds_{EMB_TYPE}.pth"))

    num_labels_task = get_num_labels(args_train.labels_task_path)
    num_labels_protected = get_num_labels(args_train.labels_protected_path)

    emb_size = train_embeddings_ds.tensors[0].shape[1]

    train_loader = DataLoader(train_embeddings_ds, shuffle=True, batch_size=128, drop_last=False)
    val_loader = DataLoader(val_embeddings_ds, shuffle=False, batch_size=128, drop_last=False)

    task_head = ClfHead(
        hid_sizes=[emb_size]*(args_train.task_n_hidden+1),
        num_labels=num_labels_task,
        dropout=args_train.task_dropout
    )
    adv_head = AdvHead(
        adv_count=args_train.adv_count,
        hid_sizes=[emb_size]*(args_train.adv_n_hidden+1),
        num_labels=num_labels_protected,
        dropout=adv_dropout
    )

    for p in task_head.parameters():
        p.requires_grad = False

    linear_adv_model = LinearAdvModel(emb_size, task_head, adv_head)

    hparams = {
        "lr": 1e-4,
        "num_epochs": 50
    }
    hparams = argparse.Namespace(**hparams)

    optimizer = SGD(linear_adv_model.parameters(), lr=hparams.lr)

    val_losses = []
    train_str = "Epoch {}, val loss: {:7.5f}"
    train_iterator = trange(hparams.num_epochs, desc=train_str.format(0, math.nan), leave=False, position=0)
    for epoch in train_iterator:

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, (X, Y) in enumerate(epoch_iterator):

            llayer.train()

            outputs = llayer(X)
            loss = loss_fn(outputs, Y)
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            llayer.zero_grad()

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

        val_iterator = tqdm(val_loader, desc=f"evaluating", leave=False, position=1)
        llayer.eval()
        for step, (X, Y) in enumerate(val_iterator):
            loss_val = []
            with torch.no_grad():
                outputs = llayer(X)
                loss_val.append(loss_fn(outputs, Y))
            loss_val = torch.cat(loss_val).mean()
            val_losses.append(loss_val.item())

        train_iterator.set_description(train_str.format(epoch, loss_val.item()), refresh=True)

    with torch.no_grad():
        y_hat_train = llayer(train_embeddings_biased)
        y_hat_val = llayer(val_embeddings_biased)

    loss_train_sgd = loss_fn(y_hat_train, train_embeddings_debiased)
    loss_val_sgd = loss_fn(y_hat_val, val_embeddings_debiased)

    print(f"train loss sgd: {loss_train_sgd.mean().item():.3f}")
    print(f"val loss sgd: {loss_val_sgd.mean().item():.3f}")

    import IPython; IPython.embed(); exit(1)


if __name__ == "__main__":

    main()