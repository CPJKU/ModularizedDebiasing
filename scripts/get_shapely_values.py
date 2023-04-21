import sys
sys.path.insert(0,'..')

import os
import ruamel.yaml as yaml
import pickle
import random
import argparse
import shap
import transformers
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import nlp
import torch
from tqdm import tqdm
from functools import partial
import pickle

from src.data_handler import get_data_loader, get_num_labels
from src.model_functions import model_factory
from src.utils import dict_to_device, get_callables
from src.adv_attack import adv_attack
from src.training_logger import TrainLogger


random.seed(0)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="dataset")
    parser.add_argument("--pk", type=str, help="protected key to create dataset for")
    args = parser.parse_args()

    with open("../cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{args.ds}"
    args = argparse.Namespace(**vars(args), **cfg["train_config"], **cfg[data_cfg], **cfg["model_config"])

    prot_key_idx = 0
    data_file = f"../analysis/data/shap_data_{args.ds}.pkl"
    shapely_file = f"../analysis/data/shap_values_{args.ds}.pkl"

    device = torch.device("cuda:0")

    cp_path_task = "/share/home/lukash/pan16/bertbase/cp/task-baseline-bert-base-uncased-64-2e-05-seed0.pt"
    cp_path_adv = "/share/home/lukash/pan16/bertbase/cp/adverserial-diff_pruning_0.01-bert-base-uncased-64-2e-05-sp_pen1.25e-07-weighted_loss_prot-gender-seed0.pt"
    cp_path_adv_mod = "/share/home/lukash/pan16/bertbase/cp_modular/modular-diff_pruning_0.01-bert-base-uncased-64-2e-05-sp_pen1.25e-07-weighted_loss_prot-gender_age-seed0.pt"
    cp_path_adv_mod_seq = "/share/home/lukash/pan16/bertbase/cp_cp_init/adverserial-diff_pruning_0.01-bert-base-uncased-64-2e-05-sp_pen1.25e-07-cp_init-weighted_loss_prot-gender-seed0.pt"
    
    model_task = model_factory(cp_path_task)
    model_adv = model_factory(cp_path_adv, remove_parametrizations=True, debiased=True)
    model_adv_mod = model_factory(cp_path_adv_mod, remove_parametrizations=True, debiased=True, debiased_par_idx=prot_key_idx)
    model_adv_mod_seq = model_factory(cp_path_adv_mod_seq, remove_parametrizations=True, debiased=True)
    
    model_task.to(device)
    model_adv.to(device)
    model_adv_mod.to(device)
    model_adv_mod_seq.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_task.model_name)

    num_labels_prot = get_num_labels(args.labels_protected_path[prot_key_idx])
    loss_fn, pred_fn, metrics = get_callables(num_labels_prot)

    dl_tokens = get_data_loader(
        args.task_key,
        args.protected_key,
        args.text_key,
        tokenizer = tokenizer,
        data_path = args.train_pkl,
        labels_task_path = args.labels_task_path,
        labels_prot_path = args.labels_protected_path,
        batch_size = args.batch_size,
        max_length = args.tokenizer_max_length,
        shuffle = False,
        debug = False
    )

    data = []
    dl_tqdm = tqdm(dl_tokens, desc=f"generating predictions")
    with torch.no_grad():
        for i, batch in enumerate(dl_tqdm):
            inputs, task_labels, prot_labels = batch[0], batch[1], batch[2+prot_key_idx]
            inputs = dict_to_device(inputs, device)
            
            emb = model_task._forward(**inputs)
            emb_deb = model_adv._forward(**inputs)
            emb_deb_mod = model_adv_mod._forward(**inputs)
            emb_deb_mod_seq = model_adv_mod_seq._forward(**inputs)

            task_pred = model_task.task_head(emb).cpu()
            task_deb_pred = model_adv.task_head(emb_deb).cpu()
            task_deb_mod_pred = model_adv_mod.task_head(emb_deb_mod).cpu()
            task_deb_mod_seq_pred = model_adv_mod_seq.task_head(emb_deb_mod_seq).cpu()

            prot_pred = []
            for m in [model_adv, model_adv_mod, model_adv_mod_seq]:
                pred_raw = m.adv_head[prot_key_idx](emb)
                pred_raw = [x.cpu() for x in pred_raw]
                if isinstance(pred_raw, list):
                    pred, _ = torch.mode(torch.stack([pred_fn(x) for x in pred_raw]), dim=0)
                else:
                    pred = pred_fn(pred_raw)
                prot_pred.append(pred)

            data.append(
                (
                    dict_to_device(inputs, "cpu"),
                    emb.cpu(),
                    emb_deb.cpu(),
                    emb_deb_mod.cpu(),
                    emb_deb_mod_seq.cpu(),
                    task_labels,
                    task_pred,
                    task_deb_pred,
                    task_deb_mod_pred,
                    task_deb_mod_seq_pred,
                    prot_labels,
                    prot_pred[0],
                    prot_pred[1],
                    prot_pred[2]
                )
            )
        
    data_dict = dict(zip(
        [
            "inputs",
            "emb",
            "emb_deb",
            "emb_deb_mod",
            "emb_deb_mod_seq",
            "task_labels",
            "task_pred",
            "task_deb_pred",
            "task_deb_mod_pred",
            "task_deb_mod_seq_pred",
            "prot_labels",
            "prot_pred",
            "prot_mod_pred",
            "prot_mod_seq_pred"
        ],
        list(zip(*data))
    ))

    transform_dict = {
        "inputs": lambda x: torch.cat([i["input_ids"] for i in x]),
        "emb": lambda x: torch.cat(x),
        "emb_deb": lambda x: torch.cat(x),
        "emb_deb_mod": lambda x: torch.cat(x),
        "emb_deb_mod_seq": lambda x: torch.cat(x),
        "task_labels": lambda x: torch.cat(x),
        "task_pred": lambda x: torch.cat(x).squeeze(),
        "task_deb_pred":  lambda x: torch.cat(x).squeeze(),
        "task_deb_mod_pred":  lambda x: torch.cat(x).squeeze(),
        "task_deb_mod_seq_pred":  lambda x: torch.cat(x).squeeze(),
        "prot_labels":  lambda x: torch.cat(x),
        "prot_pred":  lambda x: torch.cat(x).squeeze(),
        "prot_mod_pred":  lambda x: torch.cat(x).squeeze(),
        "prot_mod_seq_pred":  lambda x: torch.cat(x).squeeze()
    }

    data_dict = {k:transform_dict[k](v) for k,v in data_dict.items()}
    # import IPython; IPython.embed(); exit(1)
    # inputs, emb, emb_deb, emb_deb_mod, task_labels, task_pred, task_deb_pred, task_deb_mod_pred, prot_labels, prot_pred, prot_pred_mod = zip(*data)

    # emb = torch.cat(emb)
    # emb_deb = torch.cat(emb_deb)
    # emb_deb_mod = torch.cat(emb_deb_mod)
    # input_ids = torch.cat([x["input_ids"] for x in inputs])
    # task_labels = torch.cat(task_labels)
    # task_pred = torch.cat(task_pred).squeeze()
    # task_deb_pred = torch.cat(task_deb_pred).squeeze()
    # task_deb_mod_pred = torch.cat(task_deb_mod_pred).squeeze()
    # prot_labels = torch.cat(prot_labels)
    # prot_pred = torch.cat(prot_pred).squeeze()
    # prot_pred_mod = torch.cat(prot_pred_mod).squeeze()

    inputs_text = []
    for i in data_dict["inputs"]:
        s = tokenizer.decode(i, skip_special_tokens=True)
        inputs_text.append(s)
    data_dict["inputs_text"] = inputs_text

    train_logger = TrainLogger(
        log_dir = args.log_dir,
        logger_name = "shap_logger",
        logging_step = 5
    )       

    attack_predictions = []
    for k in ["emb", "emb_deb", "emb_deb_mod", "emb_deb_mod_seq"]:
        e = data_dict[k]
        train_loader = DataLoader(
            TensorDataset(e, data_dict["prot_labels"]), shuffle=True, batch_size=args.batch_size, drop_last=False
        )
        val_loader = DataLoader(
            TensorDataset(e[:(args.batch_size*5)], data_dict["prot_labels"][:(args.batch_size*5)]), shuffle=True, batch_size=args.batch_size, drop_last=False
        )
        attack_head = adv_attack(
            train_loader = train_loader,
            val_loader = val_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_labels_prot,
            adv_n_hidden = args.adv_n_hidden,
            adv_count = args.adv_count,
            adv_dropout = args.adv_dropout,
            num_epochs = 40,
            lr = args.learning_rate_adv_head,
            cooldown = args.cooldown,
            create_hidden_dataloader = False,
            device = device
        )
        train_loader = DataLoader(
            TensorDataset(e, data_dict["prot_labels"]), shuffle=False, batch_size=args.batch_size, drop_last=False
        )
        attack_prot_pred = []
        dl_tqdm = tqdm(train_loader, desc=f"generating predictions attack")
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dl_tqdm):
                tmp = attack_head(inputs.to(device))
                tmp = [x.cpu() for x in tmp]
                pred, _ = torch.mode(torch.stack([pred_fn(x) for x in tmp]), dim=0)
                attack_prot_pred.append(pred)
        attack_prot_pred = torch.cat(attack_prot_pred).squeeze()
        attack_predictions.append(attack_prot_pred)
        del attack_head

    data_dict = {
        **data_dict,
        "attack_pred": attack_predictions[0],
        "attack_deb_pred": attack_predictions[1],
        "attack_deb_mod_pred": attack_predictions[2],
        "attack_deb_mod_pred_seq": attack_predictions[3]
    }

    with open(data_file, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)



    def fn(x, m):
        tmp = [tokenizer.encode(v, padding="max_length", truncation=True, max_length=args.tokenizer_max_length, return_tensors="pt").squeeze() for v in x]
        tv = torch.stack(tmp).to(device)
        with torch.no_grad():
            outputs = m.forward(input_ids=tv).detach().cpu()
        return outputs.numpy()


    explainer_task = shap.Explainer(partial(fn, m=model_task), tokenizer, output_names=["mention"])
    explainer_adv = shap.Explainer(partial(fn, m=model_adv), tokenizer, output_names=["mention"])
    explainer_adv_mod = shap.Explainer(partial(fn, m=model_adv_mod), tokenizer, output_names=["mention"])
    explainer_adv_mod_seq = shap.Explainer(partial(fn, m=model_adv_mod_seq), tokenizer, output_names=["mention"])

    # keys_to_compare = ["attack_pred", "attack_deb_pred", "attack_deb_mod_pred", "attack_deb_mod_pred_seq"]
    # compare = torch.stack([data_dict[k] for k in keys_to_compare]).sum(0)
    # f_idx = (~torch.logical_or(compare == 0, compare == len(keys_to_compare))).nonzero(as_tuple=True)[0]

    # ds = {
    #     "labels": data_dict["task_labels"][f_idx],
    #     "text": [data_dict["inputs_text"][i] for i in f_idx]
    # }

    # import IPython; IPython.embed(); exit(1)
    
    ds = {
        "labels": data_dict["task_labels"],
        "text": data_dict["inputs_text"]
    }

    shap_values_task = explainer_task(ds, fixed_context=1)
    shap_values_adv = explainer_adv(ds, fixed_context=1)
    shap_values_adv_mod = explainer_adv_mod(ds, fixed_context=1)
    shap_values_adv_mod_seq = explainer_adv_mod_seq(ds, fixed_context=1)

    with open(shapely_file, "wb") as f:
        pickle.dump([shap_values_task, shap_values_adv, shap_values_adv_mod, shap_values_adv_mod_seq], f)


    # with open('temp_task.html','w') as f:
    #     f.write(shap.plots.text(shap_values_task[4], display=False))

    # with open('temp_adv.html','w') as f:
    #     f.write(shap.plots.text(shap_values_adv[4], display=False))

if __name__ == "__main__":

    main()

