import sys
sys.path.insert(0,'..')

import os
import re
import numpy as np
import torch

from typing import List, Tuple, Optional

from src.model_functions import model_factory


def get_sparsity_for_model(model, absolute: bool = False, par_idx: Optional[int] = None):

    if absolute:
        sparsity_fn = lambda x: x
    else:
        sparsity_fn = lambda x: x[2]/x[0]

    model_dict = {}
    for n,m in model.get_encoder_base_modules(return_names=True):
        # n_p, n_p_zero, n_p_one
        if par_idx is None:
            model_dict[n] = [model._count_non_zero_params_for_module(m, idx) for idx in range(model.n_parametrizations)]
        else:
            model_dict[n] = [model._count_non_zero_params_for_module(m, par_idx)]

    # overall
    overall = np.array(list(model_dict.values())).squeeze().sum(axis=0)
    overall_sparsity = [sparsity_fn(overall)]

    model_module_dict = {}
    unique_modules = set([(x[11:] if x[:10]=="embeddings" else ".".join(x.split(".")[3:])) for x in model_dict.keys() if x!="pooler.dense"])
    for module_name in unique_modules:
        for k,v in model_dict.items():
            if k[-len(module_name):] == module_name:
                try:
                    model_module_dict[module_name] += np.array(v)
                except KeyError:
                    model_module_dict[module_name] = np.array(v)
    model_module_dict = {k:[sparsity_fn(v) for v in vals] for k,vals in model_module_dict.items()}
    model_dict = {k:[sparsity_fn(v) for v in vals] for k,vals in model_dict.items()}

    if par_idx is None:
        dicts = [model._count_non_zero_params_per_layer(idx) for idx in range(model.n_parametrizations)]
    else:
        dicts = [model._count_non_zero_params_per_layer(par_idx)]

    model_layer_dict = {}
    for d in dicts:
        for k,v in d.items():
            sparsity = sparsity_fn(v)
            try:
                model_layer_dict[k].append(sparsity)
            except KeyError:
                model_layer_dict[k] = [sparsity]

    return model_dict, model_module_dict, model_layer_dict, overall_sparsity


def get_sparsity_info(folder, experiment_name, n_seeds = 5, absolute: bool = False, par_idx: Optional[int] = None):

    model_dicts = []
    model_layer_dicts = []
    model_module_dicts = []
    overall = []
    for seed in range(n_seeds):

        filepath = os.path.join(folder, experiment_name.format(seed))
        model = model_factory(filepath)

        model_dict, model_module_dict, model_layer_dict, overall_sparsity = get_sparsity_for_model(model, absolute, par_idx)
        model_dicts.append(model_dict)
        model_module_dicts.append(model_module_dict)
        model_layer_dicts.append(model_layer_dict)
        overall.append(overall_sparsity)
        

        # model_dict = {}
        # for n,m in model.get_encoder_base_modules(return_names=True):
        #     # n_p, n_p_zero, n_p_one
        #     model_dict[n] = [model._count_non_zero_params_for_module(m, idx) for idx in range(model.n_parametrizations)]

        # model_module_dict = {}
        # unique_modules = set([(x[11:] if x[:10]=="embeddings" else x[16:]) for x in model_dict.keys() if x!="pooler.dense"])
        # for module_name in unique_modules:
        #     for k,v in model_dict.items():
        #         if k[-len(module_name):] == module_name:
        #             try:
        #                 model_module_dict[module_name] += np.array(v)
        #             except KeyError:
        #                 model_module_dict[module_name] = np.array(v)
        # model_module_dict = {k:[sparsity_fn(v) for v in vals] for k,vals in model_module_dict.items()}
        # model_dict = {k:[sparsity_fn(v) for v in vals] for k,vals in model_dict.items()}

        # model_dicts.append(model_dict)
        # model_module_dicts.append(model_module_dict)

        # dicts = [model._count_non_zero_params_per_layer(idx) for idx in range(model.n_parametrizations)]

        # merged_dict = {}
        # for d in dicts:
        #     for k,v in d.items():
        #         sparsity = sparsity_fn(v)
        #         try:
        #             merged_dict[k].append(sparsity)
        #         except KeyError:
        #             merged_dict[k] = [sparsity]
        # model_layer_dicts.append(merged_dict)

    return model_dicts, model_layer_dicts, model_module_dicts, overall


def merge_info_dicts(*dicts):

    def _merge_dict_fn(l: list):
        res = {}
        for d in l:
            for k,v in d.items():
                try:
                    res[k].extend(v)
                except KeyError:
                    res[k] = v.copy()
        return res

    return [_merge_dict_fn(ds) for ds in zip(*dicts)]


def get_viz_data(model_dicts, model_layer_dicts, model_module_dicts):

    n_layers = len(model_layer_dicts[0]) - 1 # -1 to account for embedding layer

    base_dict = {}
    for i in model_dicts[0].keys():
        ar = np.array([d[i] for d in model_dicts])
        base_dict[i] = (ar.mean(0), ar.std(0))

    layer_dict = {}
    for k in model_layer_dicts[0].keys():
        ar = np.array([d[k] for d in model_layer_dicts])
        layer_dict[k] = (ar.mean(0), ar.std(0))

    module_dict = {}
    for k in model_module_dicts[0].keys():
        ar = np.array([d[k] for d in model_module_dicts])
        module_dict[k] = (ar.mean(0), ar.std(0))

    layer_name = "encoder.layer.{}"
    layer_list = []
    for i in range(n_layers):
        n = layer_name.format(i)
        d = {}
        for k, v in base_dict.items():
            k_ = k.split(".")
            if ".".join(k_[:3]) == n:
                d[".".join(k_[3:])] = v
        layer_list.append(d)

    emb_dict = {k[11:]:v for k,v in base_dict.items() if k[:10]=="embeddings"}

    return base_dict, layer_dict, module_dict, emb_dict, layer_list


def get_nonzero_dicts(model_list: List[List[Tuple]], only_nonzero = True):

    diff_masks = list(zip(*model_list))
    diff_masks_merged = {x[0][0]: torch.stack([y[1].data.requires_grad_(False) for y in x]) for x in diff_masks}
    
    base_nonzero_dict = {}
    for k, v in diff_masks_merged.items():
        if "pooler.dense" in k:
            continue
        k = ".".join(k.split(".")[:-1])
        v_bool = v.bool()
        total = v[0].numel()

        if only_nonzero:
            x = v_bool.sum(0)
        else:
            x = torch.maximum(v_bool.sum(0), (~v_bool).sum(0))

        n_nonzero = [total]
        for i in range(len(model_list)+1):
            n_nonzero.append((x == i).sum().item())
        # n_some_nonzero = torch.logical_and((weight_sums>0), (weight_sums<5)).sum().item()
        res = np.array(n_nonzero)
        
        # merge weight and bias
        try:
            base_nonzero_dict[k] += res
        except KeyError:
            base_nonzero_dict[k] = res

    layer_module_nonzero_dict = {}
    for k,v in base_nonzero_dict.items():
        if k[:10]=="embeddings":
            n = k[11:]
            i = 0
        elif (x := re.match(r"^encoder\.layer\.\d+\.", k)):
            co = x.span()[1]
            n = k[co:]
            i = int(k[:co].split(".")[-2]) + 1
        try:
            layer_module_nonzero_dict[i][n] = v
        except KeyError:
            layer_module_nonzero_dict[i] = {n: v}

    layer_nonzero_dict = {k: np.array(list(d.values())).sum(0) for k,d in layer_module_nonzero_dict.items()}
    
    module_nonzero_dict = dict(zip(
        layer_module_nonzero_dict[1].keys(),
        [np.stack(y).sum(0) for y in zip(*[list(x.values()) for x in list(layer_module_nonzero_dict.values())[1:]])]
    ))

    return base_nonzero_dict, layer_nonzero_dict, module_nonzero_dict, layer_module_nonzero_dict