import sys
sys.path.insert(0,'..')

import os
import argparse
import math
import ruamel.yaml as yaml
import itertools
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from sklearn.linear_model import LinearRegression

from src.training_logger import TrainLogger


def map_analytical(train_ds, val_ds, loss_fn, train_logger):

    X_train, Y_train, _, _ = train_ds.tensors
    X_val, Y_val, _, _ = val_ds.tensors

    linear_model = LinearRegression(fit_intercept=True)
    linear_model.fit(X_train, Y_train)

    W = torch.tensor(linear_model.coef_)
    b = torch.tensor(linear_model.intercept_)

    Y_hat_train = X_train@W.T + b
    Y_hat_val = X_val@W.T + b

    loss_train_analytical = loss_fn(Y_hat_train, Y_train).mean().item()
    loss_val_analytical = loss_fn(Y_hat_val, Y_val).mean().item()

    train_logger.reset()
    train_logger.steps = train_logger.logging_step
    train_logger.step_loss(
        step = 0,
        loss = loss_train_analytical,
        increment_steps = False,
        suffix = "analytical"
    )
    train_logger.validation_loss(
        eval_step = 0,
        result = {"loss": loss_val_analytical},
        suffix = "analytical"
    )

    print(f"train loss analytical: {loss_train_analytical:.3f}")
    print(f"val loss analytical: {loss_val_analytical:.3f}")

    return W, b


def map_gradient_based(train_ds, val_ds, loss_fn, train_logger):

    hparams = {
        "batch_size": 128,
        "lr": 1e-4,
        "num_epochs": 50,
        "seed": 0
    }
    hparams = argparse.Namespace(**hparams)

    torch.manual_seed(hparams.seed)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=hparams.batch_size, drop_last=False)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=hparams.batch_size, drop_last=False)

    emb_size = train_ds.tensors[0].shape[1]

    llayer = torch.nn.Linear(emb_size, emb_size)
    optimizer = Adam(llayer.parameters(), lr=hparams.lr)

    train_logger.reset()

    train_str = "Epoch {}, val loss: {:7.5f}"
    train_iterator = trange(hparams.num_epochs, desc=train_str.format(0, math.nan), leave=False, position=0)
    for epoch in train_iterator:

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, (X, Y, _, _) in enumerate(epoch_iterator):

            llayer.train()

            outputs = llayer(X)
            loss = loss_fn(outputs, Y)
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            llayer.zero_grad()

            train_logger.step_loss(
                step = epoch * len(train_loader) + step,
                loss = loss.item(),
                suffix = "gradient_based"
            )

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

        val_iterator = tqdm(val_loader, desc=f"evaluating", leave=False, position=1)
        llayer.eval()
        for step, (X, Y, _, _) in enumerate(val_iterator):
            loss_val = []
            with torch.no_grad():
                outputs = llayer(X)
                loss_val.append(loss_fn(outputs, Y))
            loss_val = torch.cat(loss_val).mean()

        train_logger.validation_loss(
            eval_step = epoch,
            result = {"loss": loss_val.item()},
            suffix = "gradient_based"
        )

        train_iterator.set_description(train_str.format(epoch, loss_val.item()), refresh=True)

    with torch.no_grad():
        Y_hat_train = llayer(train_ds.tensors[0])
        Y_hat_val = llayer(val_ds.tensors[0])

    loss_train_sgd = loss_fn(Y_hat_train, train_ds.tensors[1])
    loss_val_sgd = loss_fn(Y_hat_val, val_ds.tensors[1])

    print(f"train loss sgd: {loss_train_sgd.mean().item():.3f}")
    print(f"val loss sgd: {loss_val_sgd.mean().item():.3f}")

    return llayer.weight.data, llayer.bias.data


def main(args):

    log_dir = f"../logs_embeddings/{args.ds}/{args.model_type}"
    # emb_dir_in = f"/share/home/lukash/{args.ds}/{args.model_type}/embeddings"
    emb_dir_in = f"../embeddings/{args.ds}/{args.model_type}"
    emb_dir_out = f"../embeddings/{args.ds}/{args.model_type}"
    emb_type = f"modular{args.modular}_{'baseline' if args.baseline else 'fixmask'}{'' if args.baseline else args.fixmask_pct}_seed{args.seed}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(emb_dir_in, exist_ok=True)
    os.makedirs(emb_dir_out, exist_ok=True)

    train_logger = TrainLogger(
        log_dir = log_dir,
        logger_name = "_".join([args.ds, args.model_type, emb_type]),
        logging_step = 5
    )

    train_embeddings_ds = torch.load(os.path.join(emb_dir_in, f"train_embeddings_ds_{emb_type}.pth"))
    val_embeddings_ds = torch.load(os.path.join(emb_dir_in, f"val_embeddings_ds_{emb_type}.pth"))

    loss_fn = torch.nn.MSELoss(reduction="none")

    # analytical mapping
    W_ana, b_ana = map_analytical(train_embeddings_ds, val_embeddings_ds, loss_fn, train_logger)

    # gradient based mapping
    W_grad, b_grad = map_gradient_based(train_embeddings_ds, val_embeddings_ds, loss_fn, train_logger)

    modified_train_embeddings_ds = TensorDataset(
        train_embeddings_ds.tensors[0],
        train_embeddings_ds.tensors[0]@W_ana + b_ana,
        train_embeddings_ds.tensors[2],
        train_embeddings_ds.tensors[3]
    )
    torch.save(modified_train_embeddings_ds, os.path.join(emb_dir_out, f"train_embeddings_ds_{emb_type}_modified.pth"))
    modified_val_embeddings_ds = TensorDataset(
        val_embeddings_ds.tensors[0],
        val_embeddings_ds.tensors[0]@W_ana + b_ana,
        val_embeddings_ds.tensors[2],
        val_embeddings_ds.tensors[3]
    )
    torch.save(modified_val_embeddings_ds, os.path.join(emb_dir_out, f"val_embeddings_ds_{emb_type}_modified.pth"))


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
        args_ = argparse.Namespace(**comb)
        main(args_)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bertl4", help="bertbase or bertl4")
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