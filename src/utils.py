import os
import argparse
from pathlib import Path
from functools import reduce
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from typing import Union, List, Tuple, Callable, Dict, Optional

from src.training_logger import TrainLogger
from src.metrics import accuracy


def get_mean_loss(
    outputs: Union[torch.Tensor, List[torch.Tensor]],
    labels: torch.Tensor, 
    loss_fn: Callable
) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    losses = []
    for output in outputs:
        losses.append(loss_fn(output, labels))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: Callable,
    pred_fn: Callable,
    metrics: Dict[str, Callable],
    input_idx: int = 0,
    label_idx: int = 1,
    desc: str = "",
    forward_fn: Optional[Callable] = None,
    **forward_fn_kwargs
    ) -> dict:

    model.eval()

    try:
        dev = model.device
    except AttributeError:
        dev = next(model.parameters()).device

    if forward_fn is None:
        forward_fn = lambda x: model(x, **forward_fn_kwargs)

    eval_loss = 0.
    output_list = []
    val_iterator = tqdm(val_loader, desc=f"evaluating {desc}", leave=False, position=1)
    for i, batch in enumerate(val_iterator):

        inputs, labels = batch[input_idx], batch[label_idx]
        if isinstance(inputs, dict):
            inputs = dict_to_device(inputs, dev)
        else:
            inputs = inputs.to(dev)
        logits = forward_fn(inputs)
        if isinstance(logits, list):
            eval_loss += torch.stack([loss_fn(x, labels.to(dev)) for x in logits]).mean().item()
            preds, _ = torch.mode(torch.stack([pred_fn(x.cpu()) for x in logits]), dim=0)
        else:
            eval_loss += loss_fn(logits, labels.to(dev)).item()
            preds = pred_fn(logits.cpu())
        output_list.append((
            preds,
            labels
        ))

    p, l = list(zip(*output_list))
    predictions = torch.cat(p, dim=0)
    labels = torch.cat(l, dim=0)

    result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
    result["loss"] = eval_loss / (i+1)

    return result


def get_param_from_name(
    model: torch.nn.Module,
    param_name: str
):
    return reduce(lambda a,b: getattr(a,b), [model] + param_name.split("."))


def concrete_stretched(
    alpha: torch.Tensor,
    l: Union[float, int] = -1.5,
    r: Union[float, int] = 1.5,
    deterministic: bool = False
) -> torch.Tensor:
    if not deterministic:
        u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
        u_term = u.log() - (1-u).log()
    else:
        u_term = 0.
    s = (torch.sigmoid(u_term + alpha))
    s_stretched = s*(r-l) + l
    z = s_stretched.clamp(0, 1000).clamp(-1000, 1)
    return z


def dict_to_device(d: dict, device: Union[str, torch.device]) -> dict:
    return {k:v.to(device) for k,v in d.items()}


def get_device(gpu: bool, gpu_id: Union[int, list, tuple]) -> List[torch.device]:
    if gpu and torch.cuda.is_available():
        if isinstance(gpu_id, int): gpu_id = [gpu_id]
        device = [torch.device(f"cuda:{int(i)}") for i in gpu_id]
    else:
        device = [torch.device("cpu")]
    return device


def set_num_epochs_debug(args_obj: argparse.Namespace, num: int = 1) -> argparse.Namespace:
    epoch_args = [n for n in dir(args_obj) if n[:10]=="num_epochs"]
    for epoch_arg in epoch_args:
        v = min(getattr(args_obj, epoch_arg), num)
        setattr(args_obj, epoch_arg, v)
    return args_obj


def set_dir_debug(args_obj: argparse.Namespace) -> argparse.Namespace:
    dir_list = ["output_dir", "log_dir"]
    for d in dir_list:
        v = getattr(args_obj, d)
        setattr(args_obj, d, f"DEBUG_{v}")
    return args_obj


def get_name_for_run(
    baseline: bool,
    adapter: bool,
    adv: bool,
    prot_adapter: bool,
    adapter_fusion,
    modular: bool,
    args_train: argparse.Namespace,
    cp_path: bool = False,
    prot_key_idx: int = 0,
    seed: Optional[int] = None,
    debug: bool = False,
    suffix: Optional[str] = None
):
    run_parts = ["DEBUG" if debug else None]

    if modular:
        run_parts.append("modular")
    elif adv:
        run_parts.append("adverserial")
    else:
        run_parts.append("task")

    if baseline and not adapter and not prot_adapter and not adapter_fusion:
        run_parts.append("baseline")
    elif baseline and adapter:
        run_parts.append("Adp")
    elif baseline and prot_adapter:
        run_parts.append("Adp_prot")
    elif baseline and adapter_fusion:
        run_parts.append("Adp_fusion")
    else:
        run_parts.append(
            f"diff_pruning_{args_train.fixmask_pct if args_train.num_epochs_fixmask>0 else 'no_fixmask'}"
        )

    if modular:
        run_parts.extend([
            "adv_task_head" if args_train.modular_adv_task_head else None,
            "freeze_task_head" if (not args_train.modular_adv_task_head) and args_train.modular_freeze_single_task_head else None,
            "adv_merged" if args_train.modular_adv_merged else None,
            "sparse_task" if args_train.modular_sparse_task else None,
            "merged_cutoff" if args_train.modular_merged_cutoff else None
        ])

    if isinstance(args_train.protected_key, str):
        prot_attr = args_train.protected_key
    elif prot_key_idx is not None:
        prot_attr = args_train.protected_key[prot_key_idx]
    else:
        prot_attr = "_".join(args_train.protected_key)

    run_parts.extend([
        f"bottleneck_{args_train.bottleneck_dim}" if args_train.bottleneck else None,
        args_train.model_name.split('/')[-1],
        str(args_train.batch_size),
        str(args_train.learning_rate),
        f"a_samples_{args_train.concrete_samples}" if args_train.concrete_samples > 1 else None,
        f"sp_pen{args_train.sparsity_pen}" if not baseline else None,
        "cp_init" if cp_path else None,
        "triplet_loss" if args_train.triplets_loss and (adv or modular) else None,
        "weighted_loss_prot" if args_train.weighted_loss_protected and (adv or modular) else None,
        prot_attr if (adv or modular) else None,
        f"seed{seed}" if seed is not None else None,
        suffix,
    ])
    run_name = "-".join([x for x in run_parts if x is not None])
    return run_name


def get_logger(
    baseline: bool,
    adapter: bool,
    adv: bool,
    prot_adapter: bool,
    adapter_fusion: bool,
    modular: bool,
    args_train: argparse.Namespace,
    cp_path: bool = False,
    prot_key_idx: Optional[int] = None,
    seed: Optional[int] = None,
    debug: bool = False,
    suffix: Optional[str] = None
) -> TrainLogger:

    log_dir = Path(args_train.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_name = get_name_for_run(baseline, adapter, adv, prot_adapter, adapter_fusion, modular, args_train, cp_path, prot_key_idx, seed, debug, suffix)
    return TrainLogger(
        log_dir = log_dir,
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )


def get_logger_custom(
    log_dir: Union[str, Path],
    logger_name: str,
    logging_step: int = 1
) -> TrainLogger:

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return TrainLogger(
        log_dir = log_dir,
        logger_name = logger_name,
        logging_step = logging_step
    )


def get_callables(
    num_labels: int,
    class_weights: Optional[Union[int, float, torch.tensor, list, tuple]] = None
) -> Tuple[Callable, Callable, Dict[str, Callable]]:

    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights)
        if class_weights.dim() == 0:
            class_weights = class_weights.unsqueeze(0)
        if num_labels == 1:
            class_weights = class_weights[1] if len(class_weights)==2 else class_weights[0]

    if num_labels == 1:
        loss_fn = lambda x, y: torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)(x.flatten(), y.float())
        pred_fn = lambda x: (x > 0).long()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        pred_fn = lambda x: torch.argmax(x, dim=1)
    metrics = {
        "acc": accuracy,
        "balanced_acc": lambda x, y: accuracy(x, y, balanced=True)
    }
    return loss_fn, pred_fn, metrics


def get_callables_wrapper(
    num_labels: int,
    num_labels_protected: Union[int, list, tuple],
    protected_class_weights: Optional[Union[int, list, tuple]] = None
):
    loss_fn, pred_fn, metrics = get_callables(num_labels)
    if isinstance(num_labels_protected, (list, tuple)):
        callables = []
        for n, w in zip(num_labels_protected, protected_class_weights):
            callables.append(get_callables(n, class_weights = w))
        loss_fn_protected, pred_fn_protected, metrics_protected = zip(*callables)
    else:
        loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected, class_weights = protected_class_weights)
    return loss_fn, pred_fn, metrics, loss_fn_protected, pred_fn_protected, metrics_protected


def set_optional_args(args_obj: argparse.Namespace, optional_args: list) -> argparse.Namespace:
    ignored = []
    for arg in optional_args:
        assert arg.startswith("--"), "arguments need to start with '--'"
        arg_name = arg.split("=")[0][2:]
        if arg_name in args_obj:
            arg_dtype = type(getattr(args_obj, arg_name))
            if "=" in arg:
                v = arg.split("=")[1]
                arg_value = arg_dtype(v) if arg_dtype!=bool else eval(v)
            else:
                arg_value = True
            setattr(args_obj, arg_name, arg_value)
        else:
            ignored.append(arg)

    if len(ignored) > 0: print(f"ignored args: {ignored}")

    return args_obj