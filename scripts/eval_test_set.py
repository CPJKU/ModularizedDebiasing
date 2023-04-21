import sys
sys.path.insert(0,'..')

import argparse
import ruamel.yaml as yaml
import torch

from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.utils import get_logger_custom, set_optional_args, get_callables
from src.data_handler import get_data
from src.adv_attack import adv_attack


def get_model_cls(cp_name: str):
    if "modular" in cp_name:
        return ModularDiffModel
    elif "baseline" in cp_name:
        if "adv" in cp_name:
            return AdvModel
        else:
            return TaskModel
    else:
        if "adv" in cp_name:
            return AdvDiffModel
        else:
            return TaskDiffModel


GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs_test"
LOGGER_NAME = "test_logger"
# CHECKPOINTS = os.listdir("/share/home/lukash/pan16/bertl4/res_gender")
CHECKPOINTS = ["/share/home/lukash/pan16/bertl4/res_gender/cp/bert_uncased_L-4_H-256_A-4-adv_fixmask0.1-seed0.pt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
    base_args, optional = parser.parse_known_args()

    with open("../cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{base_args.ds}"
    args_train = argparse.Namespace(**cfg["train_config"], **cfg[data_cfg], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    set_optional_args(args_train, optional)

    for cp in CHECKPOINTS:

        train_logger = get_logger_custom(
            log_dir=f"../{LOG_DIR}",
            logger_name=cp
        )

        setattr(args_train, "val_pkl", args_train.test_pkl)
        train_loader, test_loader, num_labels, num_labels_protected = \
            get_data(args_train, attr_idx = 0, debug=base_args.debug) # TODO attr_idx should depend on cp name

        model_cls = get_model_cls(cp)
        model = model_cls.load_checkpoint(cp)

        loss_fn, pred_fn, metrics = get_callables(num_labels)
        loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)

        with torch.no_grad():
            if not "adv" in cp:
                result = model.evaluate(
                    test_loader,
                    loss_fn,
                    pred_fn,
                    metrics
                )
                train_logger.validation_loss(0, result, "task")

            if "adv" in cp:
                result_debiased = model.evaluate(
                    test_loader,
                    loss_fn,
                    pred_fn,
                    metrics
                )
                train_logger.validation_loss(0, result, suffix="task_debiased")

                result_protected = model.evaluate(
                    test_loader,
                    loss_fn_protected,
                    pred_fn_protected,
                    metrics_protected,
                    predict_prot=True
                )
                train_logger.validation_loss(0, result_protected, suffix="protected")
            elif "modular" in cp:
                result_debiased = model.evaluate(
                    test_loader,
                    loss_fn,
                    pred_fn,
                    metrics,
                    debiased=True
                )
                train_logger.validation_loss(0, result_debiased, suffix="task_debiased")

                result_protected = model.evaluate(
                    test_loader,
                    loss_fn_protected,
                    pred_fn_protected,
                    metrics_protected,
                    predict_prot=True,
                    debiased=True
                )
                train_logger.validation_loss(0, result_protected, suffix="protected")

        adv_attack(
            train_loader = train_loader,
            val_loader = test_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_labels_protected,
            adv_n_hidden = args_attack.adv_n_hidden,
            adv_count = args_attack.adv_count,
            adv_dropout = args_attack.adv_dropout,
            num_epochs = args_attack.num_epochs,
            lr = args_attack.learning_rate,
            cooldown = args_attack.cooldown,
            trainer = model,
            batch_size = args_attack.attack_batch_size,
            logger_suffix = f"adv_attack{'_debiased' if ('adv' in cp or 'modular' in cp) else ''}"
        )
        if "modular" in cp:
            model.set_debiased(False)
            adv_attack(
                train_loader = train_loader,
                val_loader = test_loader,
                logger = train_logger,
                loss_fn = loss_fn,
                pred_fn = pred_fn,
                metrics = metrics,
                num_labels = num_labels_protected,
                adv_n_hidden = args_attack.adv_n_hidden,
                adv_count = args_attack.adv_count,
                adv_dropout = args_attack.adv_dropout,
                num_epochs = args_attack.num_epochs,
                lr = args_attack.learning_rate,
                cooldown = args_attack.cooldown,
                trainer = model,
                batch_size = args_attack.attack_batch_size,
                logger_suffix = "adv_attack"
            )


if __name__ == "__main__":
    main()