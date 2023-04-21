from logging import LogRecord
import sys
sys.path.insert(0,'..')

import argparse
import ruamel.yaml as yaml
import torch
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_diff_adv import AdvDiffModel
from src.models.model_task import TaskModel
from src.models.model_diff_modular import ModularDiffModel
from src.models.model_heads import ClfHead
from src.models.model_base import BaseModel
from src.model_functions import train_head, merge_adv_models, merge_modular_model, merge_models
from src.adv_attack import adv_attack
from src.data_handler import get_data
from src.model_functions import generate_embeddings
from src.utils import get_logger_custom, get_callables


def merge_adv_models_wrapper(cp_gender, cp_age, cp_base = None, mean_diff_weights = True, mean_ignore_zero = False):
    model_gender = AdvDiffModel.load_checkpoint(cp_gender)
    model_age = AdvDiffModel.load_checkpoint(cp_age)

    if cp_base is not None:
        base_model = TaskModel.load_checkpoint(cp_base)
    else:
        base_model = None

    model = merge_adv_models(
        model_gender,
        model_age,
        base_model = base_model,
        mean_diff_weights = mean_diff_weights,
        mean_ignore_zero = mean_ignore_zero
    )

    # # TEMP for debugging
    # model_add = merge_adv_models(
    #     model_gender,
    #     model_age,
    #     base_model = base_model,
    #     mean_diff_weights = False
    # )
    # model_avg = merge_adv_models(
    #     model_gender,
    #     model_age,
    #     base_model = base_model,
    #     mean_diff_weights = True,
    #     mean_ignore_zero = True
    # )
    # from src.model_functions import get_param_from_name
    # base_weights = model_gender.get_base_weights(as_module=True)
    # for (n, padd), pavg in zip(model_add.named_parameters(), model_avg.parameters()):
    #     if not torch.equal(padd, pavg):
    #         n_split = n.split(".")
    #         np = ".".join(n_split[:-1] + ["parametrizations", n_split[-1], "0.diff_weight"])
    #         pg_diff = get_param_from_name(model_gender.encoder, np)
    #         pa_diff = get_param_from_name(model_age.encoder, np)
    #         p_base = get_param_from_name(base_weights, n)
    #         indices = torch.ne(padd, pavg).flatten().nonzero().flatten()
    #         i = indices[0].item()
    #         import IPython; IPython.embed(); exit(1)
    # # TEMP for debugging


    # # TEMP for debugging
    # from src.model_functions import get_param_from_name
    # from tqdm import tqdm
    # samples = []
    # with torch.no_grad():
    #     for n, _ in tqdm(list(model.named_parameters())):
    #         n_split = n.split(".")
    #         np = ".".join(n_split[:-1] + ["parametrizations", n_split[-1], "0.diff_weight"])
    #         pg_diff = get_param_from_name(model_gender.encoder, np)
    #         pa_diff = get_param_from_name(model_age.encoder, np)
    #         check = torch.logical_and(torch.logical_and(pg_diff != 0, pa_diff != 0), torch.ne(pg_diff, pa_diff)).flatten()
    #         if check.sum()>0:
    #             samples.append((n, check.nonzero()))

    # if len(samples)>0:
    #     base_weights = model_gender.get_base_weights(as_module=True)
    #     n, i = samples[0]
    #     p = get_param_from_name(model, n)
    #     np = ".".join(n.split(".")[:-1] + ["parametrizations", n.split(".")[-1], "0.diff_weight"])
    #     p_base = get_param_from_name(base_weights, ".".join(np.split(".")[1:]))
    #     pg_diff = get_param_from_name(model_gender.encoder, np)
    #     pa_diff = get_param_from_name(model_age.encoder, np)
    #     import IPython; IPython.embed(); exit(1)
    # else:
    #     print("no indices found with non zero diff weights for both models")
    # # TEMP for debugging

    return BaseModel(model_gender.model_name, model.state_dict())


def merge_modular_models_wrapper(cp_modular, mean_diff_weights = True, mean_ignore_zero = False):
    modular_model = ModularDiffModel.load_checkpoint(cp_modular)
    model = merge_modular_model(modular_model, mean_diff_weights = mean_diff_weights, mean_ignore_zero = mean_ignore_zero)
    return BaseModel(modular_model.model_name, model.state_dict())



DEBUG = False
GPU_ID = 7
SEED = 4
DS = "pan16"
PCT = 0.05
MEAN = False
MEAN_IGNORE_ZERO = True
MODULAR = True
SAME_SEED = False

torch.manual_seed(SEED)
random.seed(SEED)
 
if MODULAR:
    CP = {
        # "modular_model": f"/share/home/lukash/pan16/bertbase/cp_modular/modular-diff_pruning_{PCT}-bert-base-uncased-64-2e-05-sp_pen1.25e-07-weighted_loss_prot-gender_age-seed{SEED}.pt"
        "modular_model": f"/share/home/lukash/pan16/bertl4/cp_modular/modular-diff_pruning_{PCT}-bert_uncased_L-4_H-256_A-4-64-2e-05-weighted_loss_prot-gender_age-seed{SEED}.pt"
    }
else:
    OTHER_SEED = SEED if SAME_SEED else [x for x in range(5) if x!=0][random.randint(0,3)]
    CP = {
        "task_model": None, # f"/share/home/lukash/pan16/bertbase/cp/task-baseline-bert-base-uncased-64-2e-05-seed{SEED}.pt",
        # "adv_gender": f"/share/home/lukash/pan16/bertl4/cp_cp_init/adverserial-diff_pruning_{PCT}-sparse_task-bert_uncased_L-4_H-256_A-4-64-2e-05-sp_pen1.25e-07-cp_init-weighted_loss_prot-gender-seed{SEED}.pt",
        # "adv_age": f"/share/home/lukash/pan16/bertl4/cp_cp_init/adverserial-diff_pruning_{PCT}-sparse_task-bert_uncased_L-4_H-256_A-4-64-2e-05-sp_pen1.25e-07-cp_init-weighted_loss_prot-age-seed{OTHER_SEED}.pt"
        "adv_gender": f"/share/home/lukash/pan16/bertbase/cp_cp_init/adverserial-diff_pruning_{PCT}-sparse_task-bert-base-uncased-64-2e-05-sp_pen1.25e-07-cp_init-weighted_loss_prot-gender-seed{SEED}.pt",
        "adv_age": f"/share/home/lukash/pan16/bertbase/cp_cp_init/adverserial-diff_pruning_{PCT}-sparse_task-bert-base-uncased-64-2e-05-sp_pen1.25e-07-cp_init-weighted_loss_prot-age-seed{OTHER_SEED}.pt"
    }

LOG_DIR = f"logs_merged_masks_{DS}"

LOGGER_NAME = [
    "DEBUG" if DEBUG else None,
    "mod" if MODULAR else "adv",
    "adv_task_head" if MODULAR and ("adv_task_head" in CP["modular_model"]) else None,
    "frozen_task_head" if MODULAR and ("freeze" in CP["modular_model"]) else None,
    str(PCT),
    "avg" if MEAN else "additive",
    "ignore_zero" if MEAN and MEAN_IGNORE_ZERO else None,
    f"seed{SEED}" if MODULAR else f"seed{SEED}{OTHER_SEED}"
]
LOGGER_NAME = "_".join([x for x in LOGGER_NAME if x is not None])


DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"


def main():

    torch.manual_seed(SEED)
    print(f"torch.manual_seed({SEED})")

    if MODULAR:
        model = merge_modular_models_wrapper(
            CP["modular_model"],
            mean_diff_weights=MEAN,
            mean_ignore_zero=MEAN_IGNORE_ZERO
        )
    else:
        model = merge_adv_models_wrapper(
            CP["adv_gender"],
            CP["adv_age"],
            cp_base=CP["task_model"],
            mean_diff_weights=MEAN,
            mean_ignore_zero=MEAN_IGNORE_ZERO
        )

    model.to(DEVICE)
    model.eval()

    with open("../cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{DS}"
    args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    train_logger = get_logger_custom(
        log_dir=f"../{LOG_DIR}",
        logger_name=LOGGER_NAME
    )

    train_loader, val_loader, num_labels, num_labels_protected_list, protected_key_list, protected_class_weights_list = get_data(
        args_train = args_train,
        use_all_attr = True,
        compute_class_weights = args_train.weighted_loss_protected,
        device = DEVICE,
        debug = DEBUG
    )

    train_data = generate_embeddings(model, train_loader, forward_fn = lambda m, x: m._forward(**x))
    val_data = generate_embeddings(model, val_loader, forward_fn = lambda m, x: m._forward(**x))

    # Training Loop for task
    task_head = ClfHead(
        hid_sizes=[model.in_size_heads]*(args_train.task_n_hidden+1),
        num_labels=num_labels,
        dropout=args_train.task_dropout
    )

    # free up memory
    del model

    task_head.to(DEVICE)

    ds_train = TensorDataset(train_data[0], train_data[1])
    ds_val = TensorDataset(val_data[0], val_data[1])
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False)
    val_loader = DataLoader(ds_val, shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False)

    loss_fn, pred_fn, metrics = get_callables(num_labels)

    task_head = train_head(
        head = task_head,
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        optim = AdamW,
        num_epochs = args_attack.num_epochs,
        lr = args_attack.learning_rate,
        cooldown = args_attack.cooldown,
        desc = "task_eval"
    )

    for i, (num_lbl_prot, prot_k, prot_w) in enumerate(zip(num_labels_protected_list, protected_key_list, protected_class_weights_list)):

        label_idx = i+2

        ds_train = TensorDataset(train_data[0], train_data[label_idx])
        ds_val = TensorDataset(val_data[0], val_data[label_idx])
        train_loader = DataLoader(ds_train, shuffle=True, batch_size=args_attack.attack_batch_size, drop_last=False)
        val_loader = DataLoader(ds_val, shuffle=False, batch_size=args_attack.attack_batch_size, drop_last=False)

        loss_fn, pred_fn, metrics = get_callables(num_lbl_prot, prot_w)

        adv_attack(
            train_loader = train_loader,
            val_loader = val_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_lbl_prot,
            adv_n_hidden = args_attack.adv_n_hidden,
            adv_count = args_attack.adv_count,
            adv_dropout = args_attack.adv_dropout,
            num_epochs = args_attack.num_epochs,
            lr = args_attack.learning_rate,
            cooldown = args_attack.cooldown,
            create_hidden_dataloader = False,
            device = DEVICE,
            logger_suffix = f"adv_attack_{prot_k}"
        )


if __name__ == "__main__":
    main()