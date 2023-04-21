import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import OrderedDict

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import ClfHead, AdvHead
from src.models.model_base import BasePruningModel
from src.models.model_adv import AdvModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device, evaluate_model, get_mean_loss


class AdvDiffModel(BasePruningModel):

    def __init__(
        self,
        model_name: str,
        num_labels_task: int,
        num_labels_protected: Union[int, list, tuple],
        task_dropout: float = .3,
        task_n_hidden: int = 0,
        adv_dropout: float = .3,
        adv_n_hidden: int = 1,
        adv_count: int = 5,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        fixmask_init: bool = False,
        concrete_lower: Optional[float] = -1.5,
        concrete_upper: Optional[float] = 1.5,
        structured_diff_pruning: Optional[bool] = True,
        alpha_init: Optional[Union[int, float]] = 5,
        encoder_state_dict: OrderedDict = None,
        state_dict_load_to_par: bool = False,
        task_head_state_dict: OrderedDict = None,
        task_head_freeze: bool = False,
        **kwargs
    ):
        if state_dict_load_to_par:
            super().__init__(model_name, **kwargs)
        else:
            super().__init__(model_name, encoder_state_dict=encoder_state_dict, **kwargs)  
        
        if isinstance(num_labels_protected, int):
            num_labels_protected = [num_labels_protected]

        self.num_labels_task = num_labels_task
        self.num_labels_protected = num_labels_protected
        self.task_dropout = task_dropout
        self.task_n_hidden = task_n_hidden
        self.adv_dropout = adv_dropout
        self.adv_n_hidden = adv_n_hidden
        self.adv_count = adv_count
        self.has_bottleneck = bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_dropout = bottleneck_dropout
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.structured_diff_pruning = structured_diff_pruning
        self.task_head_freeze = task_head_freeze

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = torch.nn.Identity()
            self.in_size_heads = self.hidden_size

        # heads
        self.task_head = ClfHead([self.in_size_heads]*(task_n_hidden+1), num_labels_task, dropout=task_dropout)
        if task_head_state_dict is not None:
            self.task_head.load_state_dict(task_head_state_dict)
            if task_head_freeze:
                for p in self.task_head.parameters():
                    p.requires_grad = False

        self.adv_head = torch.nn.ModuleList()
        for n in num_labels_protected:
            self.adv_head.append(
                AdvHead(adv_count, hid_sizes=[self.in_size_heads]*(adv_n_hidden+1), num_labels=n, dropout=adv_dropout)
            )

        self._add_diff_parametrizations(
            n_parametrizations = 1,
            p_requires_grad = False,
            fixmask_init = fixmask_init,
            alpha_init = alpha_init,
            concrete_lower = concrete_lower,
            concrete_upper = concrete_upper,
            structured = structured_diff_pruning
        )

        if encoder_state_dict is not None and state_dict_load_to_par:
            self.load_state_dict_to_parametrizations(encoder_state_dict)


    def _forward(self, **x) -> torch.Tensor:
        hidden = super()._forward(**x)
        return self.bottleneck(hidden)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))

    def forward_protected(self, head_idx=0, **x) -> torch.Tensor:
        return self.adv_head[head_idx](self._forward(**x))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        loss_fn_protected: Union[Callable, list, tuple],
        pred_fn_protected: Union[Callable, list, tuple],
        metrics_protected: Union[Dict[str, Callable], list, tuple],
        num_epochs_warmup: int,
        num_epochs_finetune: int,
        num_epochs_fixmask: int,
        concrete_samples: int,
        adv_lambda: float,
        sparsity_pen: Union[float, list, tuple],
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        learning_rate_alpha: float,
        optimizer_warmup_steps: int,
        weight_decay: float,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        concrete_lower: Optional[float] = None,
        concrete_upper: Optional[float] = None,
        structured_diff_pruning: Optional[bool] = None,
        fixmask_pct: Optional[float] = None,
        protected_key: Optional[Union[str, list, tuple]] = None,
        checkpoint_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:

        assert self.finetune_state or (self.fixmask_state and num_epochs_finetune==0), \
            "model is in fixmask state but num_epochs_fintune>0"

        if not isinstance(loss_fn_protected, (list, tuple)):
            loss_fn_protected = [loss_fn_protected]
        if not isinstance(pred_fn_protected, (list, tuple)):
            pred_fn_protected = [pred_fn_protected]
        if not isinstance(metrics_protected, (list, tuple)):
            metrics_protected = [metrics_protected]
        if not isinstance(protected_key, (list, tuple)):
            protected_key = [protected_key]
        if protected_key[0] is None:
            protected_key = list(range(len(protected_key)))

        self.global_step = 0
        num_epochs_finetune += num_epochs_warmup
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) * num_epochs_finetune
        train_steps_fixmask = len(train_loader) * num_epochs_fixmask

        par_attr = {
            "concrete_lower": concrete_lower,
            "concrete_upper": concrete_upper,
            "structured_diff_pruning": structured_diff_pruning
        }
        for s, v in par_attr.items():
            if v is not None:
                setattr(self, s, v)
                self._parametrizations_set_attr(s, v)

        log_ratio = self.get_log_ratio(self.concrete_lower, self.concrete_upper)
        sparsity_pen = self.get_sparsity_pen(sparsity_pen)

        self._init_optimizer_and_schedule(
            train_steps_finetune,
            learning_rate,
            learning_rate_task_head,
            learning_rate_adv_head,
            learning_rate_alpha,
            learning_rate_bottleneck,
            weight_decay,
            optimizer_warmup_steps
        )

        train_str = "Epoch {}, model_state: {}, {}"
        str_suffix = lambda d, suffix="": ", ".join([f"{k}{suffix}: {v}" for k,v in d.items()])

        train_iterator = trange(num_epochs_total, desc=train_str.format(0, self.model_state, ""), leave=False, position=0)
        for epoch in train_iterator:

            if epoch<num_epochs_warmup:
                _adv_lambda = 0.
            else:
                _adv_lambda = adv_lambda

            # switch to fixed mask training
            if epoch == num_epochs_finetune:
                self._finetune_to_fixmask(fixmask_pct)
                self._init_optimizer_and_schedule(
                    train_steps_fixmask,
                    learning_rate,
                    learning_rate_task_head,
                    learning_rate_adv_head,
                    learning_rate_alpha,
                    learning_rate_bottleneck,
                    weight_decay,
                    optimizer_warmup_steps
                )

            self._step(
                train_loader,
                loss_fn,
                logger,
                log_ratio,
                sparsity_pen,
                max_grad_norm,
                loss_fn_protected,
                _adv_lambda,
                concrete_samples
            )

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics
            )
            logger.validation_loss(epoch, result, suffix="task_debiased")

            results_protected = {}
            for i, (prot_key, loss_fn_prot, pred_fn_prot, metrics_prot) in enumerate(zip(
                protected_key, loss_fn_protected, pred_fn_protected, metrics_protected
            )):
                k = f"protected_{prot_key}"
                res_prot = self.evaluate(
                    val_loader,
                    loss_fn_prot,
                    pred_fn_prot,
                    metrics_prot,
                    label_idx=i+2
                )
                results_protected[k] = res_prot
                logger.validation_loss(epoch, res_prot, suffix=k)

            # count non zero
            n_p, n_p_zero, n_p_one = self._count_non_zero_params()
            n_p_between = n_p - (n_p_zero + n_p_one)
            logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between, "adv")

            result_name = "_task_debiased" if len(protected_key)>1 else f"_task_debiased_{protected_key[0]}"
            result_strings = [str_suffix(result, result_name)]
            for k, r in results_protected.items():
                result_strings.append(str_suffix(r, f"_{k}"))
            result_str = ", ".join(result_strings)

            train_iterator.set_description(
                train_str.format(epoch, self.model_state, result_str), refresh=True
            )

            if self.fixmask_state or ((num_epochs_fixmask == 0) and (epoch >= num_epochs_warmup)):
                cpt = self.save_checkpoint(
                    Path(output_dir),
                    checkpoint_name,
                    seed
                )

        print("Final result after " + train_str.format(epoch, self.model_state, result_str))

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        label_idx: int = 1
    ) -> dict:
        self.eval()

        if label_idx > 1:
            desc = f"protected attribute {label_idx-2}"
            forward_fn = lambda x: self.forward_protected(head_idx=label_idx-2, **x)
        else:
            desc = "task"
            forward_fn = lambda x: self(**x)
            label_idx = max(1, label_idx)

        return evaluate_model(
            self,
            val_loader,
            loss_fn,
            pred_fn,
            metrics,
            input_idx=0,
            label_idx=label_idx,
            desc=desc,
            forward_fn=forward_fn
        )


    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        log_ratio: float,
        sparsity_pen: list,
        max_grad_norm: float,
        loss_fn_protected: Union[list, tuple],
        adv_lambda: float,
        concrete_samples: int
    ) -> None:

        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}, loss l0 pen: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels_task = batch[:2]
            labels_protected = batch[2:]
            inputs = dict_to_device(inputs, self.device)

            concrete_samples = concrete_samples if self.finetune_state else 1

            loss = 0.
            partial_losses = torch.zeros((3,))
            for _ in range(concrete_samples):

                hidden = self._forward(**inputs)
                outputs_task = self.task_head(hidden)
                loss_task = loss_fn(outputs_task, labels_task.to(self.device))
                loss += loss_task

                for i, (l, loss_fn_prot) in enumerate(zip(labels_protected, loss_fn_protected)):
                    outputs_protected = self.adv_head[i].forward_reverse(hidden, lmbda = adv_lambda)
                    loss_protected = get_mean_loss(outputs_protected, l.to(self.device), loss_fn_prot)
                    loss += loss_protected

                loss_l0 = torch.tensor(0.)
                if self.finetune_state:
                    loss_l0 = self._get_sparsity_loss(log_ratio, sparsity_pen, 0)
                    loss += loss_l0

                partial_losses += torch.tensor([loss_task, loss_protected, loss_l0]).detach()

            loss /= concrete_samples
            partial_losses /= concrete_samples

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

            self.optimizer.step()
            # self.scheduler.step()
            self.zero_grad()

            losses_dict = {
                "total_adv": loss.item(),
                "task_adv": partial_losses[0],
                "protected": partial_losses[1],
                "l0_adv": partial_losses[2]
            }
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(step, loss.item(), partial_losses[2]), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_task_head: float,
        learning_rate_adv_head: float,
        learning_rate_alpha: float,
        learning_rate_bottleneck: float = 1e-4,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 0
    ) -> None:

        optimizer_param_groups = [
            {
                "params": self.bottleneck.parameters(),
                "lr": learning_rate_bottleneck
            },
            {
                "params": self.adv_head.parameters(),
                "lr": learning_rate_adv_head
            }
        ]

        if not self.task_head_freeze:
            optimizer_param_groups.append({
                "params": self.task_head.parameters(),
                "lr": learning_rate_task_head
            })

        optimizer_param_groups.extend(
            self._get_diff_param_groups(learning_rate, weight_decay, learning_rate_alpha, 0)
        )

        self.optimizer = AdamW(optimizer_param_groups, betas=(0.9, 0.999), eps=1e-08)

        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        # )


    def make_checkpoint_name(
        self,
        seed: Optional[int] = None
    ):
        filename_parts = [
            self.model_name.split('/')[-1],
            "adv_" + f"fixmask{self.fixmask_pct}" if self.fixmask_state else "diff_pruning",
            "cp_init" if self.state_dict_init else None,
            f"seed{seed}" if seed is not None else None
        ]
        return "-".join([x for x in filename_parts if x is not None]) + ".pt"


    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        checkpoint_name: Optional[str] = None,
        seed: Optional[None] = None
    ) -> None:
        info_dict = {
            "cls_name": self.__class__.__name__,
            "model_name": self.model_name,
            "num_labels_task": self.num_labels_task,
            "num_labels_protected": self.num_labels_protected,
            "task_dropout": self.task_dropout,
            "task_n_hidden": self.task_n_hidden,
            "adv_dropout": self.adv_dropout,
            "adv_n_hidden": self.adv_n_hidden,
            "adv_count": self.adv_count,
            "bottleneck": self.has_bottleneck,
            "bottleneck_dim": self.bottleneck_dim,
            "bottleneck_dropout": self.bottleneck_dropout,
            "fixmask": self.fixmask_state,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "adv_head_state_dict": self.adv_head.state_dict(),
            "concrete_lower": self.concrete_lower,
            "concrete_upper": self.concrete_upper,
            "structured_diff_pruning": self.structured_diff_pruning,
            "task_head_freeze": self.task_head_freeze
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_name is None:
            checkpoint_name = self.make_checkpoint_name(seed)
        filepath = output_dir / checkpoint_name
        torch.save(info_dict, filepath)
        return filepath


    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, os.PathLike],
        remove_parametrizations: bool = False,
        map_location: Union[str, torch.device] = torch.device('cpu')
    ) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)

        cls_instance = cls(
            model_name = info_dict['model_name'],
            num_labels_task = info_dict['num_labels_task'],
            num_labels_protected = info_dict['num_labels_protected'],
            task_dropout = info_dict['task_dropout'],
            task_n_hidden = info_dict['task_n_hidden'],
            adv_dropout = info_dict['adv_dropout'],
            adv_n_hidden = info_dict['adv_n_hidden'],
            adv_count = info_dict['adv_count'],
            bottleneck = info_dict['bottleneck'],
            bottleneck_dim = info_dict['bottleneck_dim'],
            bottleneck_dropout = info_dict['bottleneck_dropout'],
            fixmask_init = info_dict['fixmask'],
            concrete_lower = info_dict['concrete_lower'],
            concrete_upper = info_dict['concrete_upper'],
            structured_diff_pruning = info_dict['structured_diff_pruning'],
            task_head_freeze = info_dict['task_head_freeze']
        )

        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])

        if remove_parametrizations:

            cls_instance._remove_parametrizations()
            
            unparametrized_model = AdvModel(
                model_name = info_dict['model_name'],
                num_labels_task = info_dict['num_labels_task'],
                num_labels_protected = info_dict['num_labels_protected'],
                task_dropout = info_dict['task_dropout'],
                task_n_hidden = info_dict['task_n_hidden'],
                adv_dropout = info_dict['adv_dropout'],
                adv_n_hidden = info_dict['adv_n_hidden'],
                adv_count = info_dict['adv_count'],
                bottleneck = info_dict['bottleneck'],
                bottleneck_dim = info_dict['bottleneck_dim'],
                bottleneck_dropout = info_dict['bottleneck_dropout']                 
            )
            unparametrized_model.encoder.load_state_dict(cls_instance.encoder.state_dict())
            unparametrized_model.task_head.load_state_dict(cls_instance.task_head.state_dict())
            unparametrized_model.adv_head.load_state_dict(cls_instance.adv_head.state_dict())

            unparametrized_model.eval()
            return unparametrized_model

        else:

            cls_instance.eval()
            return cls_instance


