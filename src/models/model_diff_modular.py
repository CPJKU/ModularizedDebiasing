import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import AdvHead, ClfHead
from src.models.model_base import BasePruningModel
from src.models.model_task import TaskModel
from src.models.model_adv import AdvModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device, evaluate_model, get_mean_loss


class ModularDiffModel(BasePruningModel):

    @property
    def n_embeddings(self):
        return (not self.sparse_task) + self.n_parametrizations

    @property
    def _task_head_idx(self):
        return self.adv_task_head*self._debiased*(1+self._debiased_par_idx)

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
        adv_task_head: bool = True,
        freeze_single_task_head: bool = True,
        adv_merged: bool = True,
        sparse_task: bool = False,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

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
        self.adv_task_head = adv_task_head
        self.freeze_single_task_head = freeze_single_task_head
        self.adv_merged = adv_merged
        self.sparse_task = sparse_task

        # bottleneck layer
        n_bottleneck = 1 + max(1, (not adv_merged) * len(num_labels_protected))
        if self.has_bottleneck:
            self.bottleneck = torch.nn.ModuleList([
                ClfHead(hid_sizes=self.hidden_size, num_labels=bottleneck_dim, dropout=bottleneck_dropout) for _ in range(n_bottleneck)
            ])
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n_bottleneck)])
            self.in_size_heads = self.hidden_size

        # task head
        n_task_head = 1 + adv_task_head * max(1, (not adv_merged) * len(num_labels_protected))
        self.task_head = torch.nn.ModuleList([
            ClfHead(hid_sizes=[self.in_size_heads]*(task_n_hidden+1), num_labels=num_labels_task, dropout=task_dropout) for _ in range(n_task_head)
        ])             

        # adv head
        self.adv_head = torch.nn.ModuleList([
            AdvHead(adv_count, hid_sizes=[self.in_size_heads]*(adv_n_hidden+1), num_labels=n, dropout=adv_dropout) for n in num_labels_protected
        ])

        n_parametrizations = sparse_task + max(1, (not adv_merged) * len(num_labels_protected))
        self._add_diff_parametrizations(
            n_parametrizations = n_parametrizations,
            p_requires_grad = not sparse_task,
            fixmask_init = fixmask_init,
            alpha_init = alpha_init,
            concrete_lower = concrete_lower,
            concrete_upper = concrete_upper,
            structured = structured_diff_pruning
        )

        self.set_debiased(False)


    def _forward(self, **x) -> torch.Tensor:
        hidden = super()._forward(**x)
        idx = self._debiased*(1+self._debiased_par_idx)
        return self.bottleneck[idx](hidden)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head[self._task_head_idx](self._forward(**x))

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
        merged_cutoff: bool,
        merged_min_pct: float,
        cooldown: Optional[int] = None,
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

        do_task_step = True
        performance_decrease_counter = 0

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
                sequential = [True] * self.sparse_task + [False] + [False] * (self.n_parametrizations-self.sparse_task-1) * (not self.adv_merged)
                self._finetune_to_fixmask(
                    fixmask_pct,
                    sequential = sequential,
                    merged_cutoff = merged_cutoff,
                    merged_min_pct = merged_min_pct
                )
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
                concrete_samples,
                do_task_step
            )

            results_task = {}
            for i in range(self.n_embeddings):
                k_debiased_idx = max(0,i-1)
                res_task = self.evaluate(
                    val_loader,
                    loss_fn,
                    pred_fn,
                    metrics,
                    label_idx=1,
                    debiased=bool(i),
                    debiased_par_idx=k_debiased_idx
                )
                if i == 0:
                    k = "task"
                elif i > 0 and self.adv_merged:
                    k = "task_debiased"
                else:
                    k = f"task_debiased_{protected_key[k_debiased_idx]}"
                results_task[k] = res_task
                logger.validation_loss(epoch, res_task, suffix=k)

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
                    label_idx=i+2,
                    debiased=True,
                    debiased_par_idx=i
                )
                results_protected[k] = res_prot
                logger.validation_loss(epoch, res_prot, suffix=k)

            # count non zero
            mask_names = ["task"]
            if self.adv_merged:
                mask_names.append("adv")
            else:
                mask_names.extend([f"adv_{k}" for k in protected_key])           
            for i, suffix in enumerate(mask_names[(not self.sparse_task):]):
                n_p, n_p_zero, n_p_one = self._count_non_zero_params(idx = i)
                n_p_between = n_p - (n_p_zero + n_p_one)
                logger.non_zero_params(epoch, n_p, n_p_zero, n_p_between, suffix)

            result_strings = []
            for k, r in results_task.items():
                result_strings.append(str_suffix(r, f"_{k}"))
            for k, r in results_protected.items():
                result_strings.append(str_suffix(r, f"_{k}"))
            result_str = ", ".join(result_strings)

            train_iterator.set_description(
                train_str.format(epoch, self.model_state, result_str), refresh=True
            )

            check_save_cp = self.fixmask_state or ((num_epochs_fixmask == 0) and (epoch >= num_epochs_warmup))
            check_do_task_step = ((self.sparse_task and check_save_cp) or (not self.sparse_task)) and \
                (cooldown is not None) and \
                (self.adv_task_head or ((not self.adv_task_head) and self.freeze_single_task_head))

            if check_do_task_step:   
                if logger.is_best(results_task["task"]["loss"], ascending=True, id="loss"):
                    performance_decrease_counter = 0
                else:
                    performance_decrease_counter += 1
                do_task_step = (performance_decrease_counter <= cooldown)

            if check_save_cp:
                cpt = self.save_checkpoint(
                    Path(output_dir),
                    checkpoint_name,
                    seed
                )

            # TEMP
            logger.validation_loss(epoch, {"do task step": int(do_task_step)})

        self.set_debiased(False) # deactivate debiasing at end of training

        print("Final results after " + train_str.format(epoch, self.model_state, result_str))

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        label_idx: int = 1,
        debiased: bool = False,
        debiased_par_idx: int = 0
    ) -> dict:
        self.eval()

        if label_idx > 1:
            desc = f"protected attribute {label_idx-2}"
            forward_fn = lambda x: self.forward_protected(head_idx=label_idx-2, **x)
        else:
            desc = "task"
            forward_fn = lambda x: self(**x)
            label_idx = max(1, label_idx)

        debiased_before = self._debiased
        idx_before = self._debiased_par_idx
        self.set_debiased(debiased, grad_switch=False, debiased_par_idx=debiased_par_idx)
        result = evaluate_model(
            self,
            val_loader,
            loss_fn,
            pred_fn,
            metrics,
            label_idx=label_idx,
            desc=desc,
            forward_fn=forward_fn
        )
        self.set_debiased(debiased_before, grad_switch=False, debiased_par_idx=idx_before)

        return result


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
        concrete_samples: int,
        do_task_step: bool = True
    ) -> None:

        self.train()

        epoch_str = "training - step {}, loss_biased: {:7.5f}, loss_debiased: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels_task = batch[:2]
            labels_protected = batch[2:]
            inputs = dict_to_device(inputs, self.device)

            concrete_samples = concrete_samples if self.finetune_state else 1
            concrete_samples_task = concrete_samples if self.sparse_task else 1

            ##################################################
            # START STEP TASK
            loss_biased = torch.tensor(0., device=self.device)
            partial_losses_biased = torch.zeros((2,))

            if do_task_step:

                self.set_debiased(False)

                for _ in range(concrete_samples_task):

                    outputs = self(**inputs)
                    loss_task = loss_fn(outputs, labels_task.to(self.device))
                    loss_biased += loss_task

                    loss_l0 = torch.tensor(0.)
                    if self.finetune_state and self.sparse_task:
                        loss_l0 = self._get_sparsity_loss(log_ratio, sparsity_pen, 0)
                        loss_biased += loss_l0

                    partial_losses_biased += torch.tensor([loss_task, loss_l0]).detach()

                loss_biased /= concrete_samples_task
                partial_losses_biased /= concrete_samples_task

                loss_biased.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

                self.optimizer_task.step()
                self.optimizer_task.zero_grad()

            # pnames, params = zip(*[(n,p) for n,p in self.encoder.named_parameters() if n[-9:] == f".original"])
            # rand_idx = [torch.randperm(p.numel())[0].item() for p in params]
            # rand_params = torch.stack([p.view(-1)[i].detach() for p,i in zip(params, rand_idx)])

            # END STEP TASK
            ##################################################

            ##################################################
            # START STEP DEBIAS

            if self.adv_merged:

                self.set_debiased(True)    

                loss_debiased = 0.
                partial_losses_debiased = torch.zeros((3,))

                for _ in range(concrete_samples):

                    hidden = self._forward(**inputs)

                    outputs_task = self.task_head[self._task_head_idx](hidden)
                    loss_task_adv = loss_fn(outputs_task, labels_task.to(self.device))
                    loss_debiased += loss_task_adv

                    loss_protected = 0.
                    for i, (l, loss_fn_prot) in enumerate(zip(labels_protected, loss_fn_protected)):
                        outputs_protected = self.adv_head[i].forward_reverse(hidden, lmbda = adv_lambda)
                        loss_protected += get_mean_loss(outputs_protected, l.to(self.device), loss_fn_prot)
                        loss_debiased += loss_protected

                    loss_l0_adv = torch.tensor(0.)
                    if self.finetune_state:
                        loss_l0_adv = self._get_sparsity_loss(log_ratio, sparsity_pen, int(self.sparse_task))
                        loss_debiased += loss_l0_adv   

                    partial_losses_debiased += torch.tensor([loss_task_adv, loss_protected, loss_l0_adv]).detach()

                loss_debiased /= concrete_samples
                partial_losses_debiased /= concrete_samples

                loss_debiased.backward()

                losses_debiased_dict = {
                    "total_adv": loss_debiased.item(),
                    "task_adv": partial_losses_debiased[0],
                    "protected": partial_losses_debiased[1],
                    "l0_adv": partial_losses_debiased[2]
                }

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

                self.optimizers_adv[0].step()
                self.optimizers_adv[0].zero_grad()

            else:     

                losses_debiased_dict = {}   

                for debiased_par_idx in range(self.n_parametrizations-self.sparse_task):

                    self.set_debiased(True, debiased_par_idx=debiased_par_idx)

                    loss_debiased = 0.
                    partial_losses_debiased = torch.zeros((3,))

                    for _ in range(concrete_samples):

                        hidden = self._forward(**inputs)

                        outputs_task = self.task_head[self._task_head_idx](hidden)
                        loss_task_adv = loss_fn(outputs_task, labels_task.to(self.device))
                        loss_debiased += loss_task_adv

                        l = labels_protected[debiased_par_idx]
                        loss_fn_prot = loss_fn_protected[debiased_par_idx]
                        outputs_protected = self.adv_head[debiased_par_idx].forward_reverse(hidden, lmbda = adv_lambda)
                        loss_protected = get_mean_loss(outputs_protected, l.to(self.device), loss_fn_prot)
                        loss_debiased += loss_protected

                        loss_l0_adv = torch.tensor(0.)
                        if self.finetune_state:
                            loss_l0_adv = self._get_sparsity_loss(log_ratio, sparsity_pen, int(self.sparse_task))
                            loss_debiased += loss_l0_adv

                        partial_losses_debiased += torch.tensor([loss_task_adv, loss_protected, loss_l0_adv]).detach()

                    loss_debiased /= concrete_samples
                    partial_losses_debiased /= concrete_samples

                    loss_debiased.backward()

                    losses_debiased_dict = {
                        **losses_debiased_dict,
                        f"total_adv_{debiased_par_idx}": loss_debiased.item(),
                        f"task_adv_{debiased_par_idx}": partial_losses_debiased[0],
                        f"protected_{debiased_par_idx}": partial_losses_debiased[1],
                        f"l0_adv_{debiased_par_idx}": partial_losses_debiased[2]
                    }

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

                    self.optimizers_adv[debiased_par_idx].step()
                    self.optimizers_adv[debiased_par_idx].zero_grad()

            # pnames_after, params_after = zip(*[(n,p) for n,p in self.encoder.named_parameters() if n[-9:] == f".original"])
            # rand_params_after = torch.stack([p.view(-1)[i].detach() for p,i in zip(params_after, rand_idx)])
            #assert torch.equal(rand_params, rand_params_after)

            # END STEP DEBIAS
            ##################################################

            # self.scheduler.step()

            losses_dict = {
                "total": loss_biased.item() + loss_debiased.item(),
                "total_biased": loss_biased.item(),
                "task": partial_losses_biased[0],
                "l0": partial_losses_biased[1],
                **losses_debiased_dict
            }
            logger.step_loss(self.global_step, losses_dict)

            epoch_iterator.set_description(epoch_str.format(step, loss_biased.item(), loss_debiased.item()), refresh=True)

            self.global_step += 1


    def set_debiased(self, debiased: bool, grad_switch: bool = True, debiased_par_idx: int = 0) -> None:
        try:
            check_debiased = (debiased != self._debiased)
        except AttributeError:
            check_debiased = True

        debiased_par_idx *= (not self.adv_merged)
        try:
            check_idx = (debiased_par_idx != self._debiased_par_idx)
        except AttributeError:
            check_idx = True

        if check_debiased and grad_switch:
            if (not self.adv_task_head) and self.freeze_single_task_head:
                self.task_head[0].freeze_parameters(frozen=debiased)
            if self.sparse_task:
                self._freeze_parametrizations(debiased, 0)
            else:
                self._freeze_original_parameters(debiased)
        if check_debiased or check_idx:
            for i in range(self.n_parametrizations-self.sparse_task):
                active = bool(debiased * (i==debiased_par_idx))
                self._activate_parametrizations(active, idx=i)
            self._debiased = debiased
            self._debiased_par_idx = debiased_par_idx


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

        # task params
        task_optimizer_param_groups = [
            {
                "params": self.bottleneck[0].parameters(),
                "lr": learning_rate_bottleneck
            },
            {
                "params": self.task_head[0].parameters(),
                "lr": learning_rate_task_head
            }
        ]

        if self.sparse_task:
            task_optimizer_param_groups.extend(
                self._get_diff_param_groups(learning_rate, weight_decay, learning_rate_alpha, 0)
            )
        else:
            task_optimizer_param_groups.append(
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-9:] == f".original"],
                    "lr": learning_rate,
                    "weight_decay": weight_decay
                }
            )

        self.optimizer_task = AdamW(task_optimizer_param_groups, betas=(0.9, 0.999), eps=1e-08)

        # adv params
        self.optimizers_adv = []
        
        for i in range(self.sparse_task, self.n_parametrizations):
            
            adv_optimizer_param_groups = [
                {
                    "params": self.bottleneck[i].parameters(),
                    "lr": learning_rate_bottleneck
                }
            ]

            if self.adv_merged:
                adv_optimizer_param_groups.append({
                    "params": self.adv_head.parameters(),
                    "lr": learning_rate_adv_head
                })
            else:
                adv_optimizer_param_groups.append({
                    "params": self.adv_head[i-self.sparse_task].parameters(),
                    "lr": learning_rate_adv_head
                })
                
            if self.adv_task_head:
                adv_optimizer_param_groups.append({
                    "params": self.task_head[i-self.sparse_task+1].parameters(),
                    "lr": learning_rate_task_head
                })
            elif not self.freeze_single_task_head:
                adv_optimizer_param_groups.append({
                    "params": self.task_head[0].parameters(),
                    "lr": learning_rate_task_head
                })

            adv_optimizer_param_groups.extend(
                self._get_diff_param_groups(learning_rate, weight_decay, learning_rate_alpha, idx=i)
            )

            self.optimizers_adv.append(
                AdamW(adv_optimizer_param_groups, betas=(0.9, 0.999), eps=1e-08)
            )

        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        # )


    def make_checkpoint_name(
        self,
        seed: Optional[int] = None
    ):
        filename_parts = [
            self.model_name.split('/')[-1],
            "modular_" + f"fixmask{self.fixmask_pct}" if self.fixmask_state else "diff_pruning",
            "sparse_task" if self.sparse_task else None,
            "merged_head" if not self.adv_task_head else None,
            f"seed{seed}" if seed is not None else None
        ]
        return "-".join([x for x in filename_parts if x is not None]) + ".pt"


    def save_checkpoint(
        self,
        output_dir: Union[str, os.PathLike],
        checkpoint_name: Optional[str] = None,
        seed: Optional[int] = None
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
            "adv_task_head": self.adv_task_head,
            "freeze_single_task_head": self.freeze_single_task_head,
            "adv_merged": self.adv_merged,
            "sparse_task": self.sparse_task
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
        debiased: bool = True,
        debiased_par_idx: int = 0,
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
            adv_task_head = info_dict['adv_task_head'],
            freeze_single_task_head = info_dict['freeze_single_task_head'],
            adv_merged = info_dict['adv_merged'],
            sparse_task = info_dict['sparse_task']
        )

        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])
        cls_instance.adv_head.load_state_dict(info_dict['adv_head_state_dict'])

        cls_instance.set_debiased(debiased, debiased_par_idx=debiased_par_idx)

        if remove_parametrizations:

            cls_instance._remove_parametrizations()
            
            if debiased:
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
                unparametrized_model.task_head.load_state_dict(cls_instance.task_head[cls_instance._task_head_idx].state_dict())
                unparametrized_model.adv_head.load_state_dict(cls_instance.adv_head.state_dict())
            else:
                unparametrized_model = TaskModel(
                    model_name = info_dict['model_name'],
                    num_labels = info_dict['num_labels_task'],
                    dropout = info_dict['task_dropout'],
                    n_hidden = info_dict['task_n_hidden'],
                    bottleneck = info_dict['bottleneck'],
                    bottleneck_dim = info_dict['bottleneck_dim'],
                    bottleneck_dropout = info_dict['bottleneck_dropout']                 
                )
                unparametrized_model.encoder.load_state_dict(cls_instance.encoder.state_dict())
                unparametrized_model.task_head.load_state_dict(cls_instance.task_head[cls_instance._task_head_idx].state_dict())

            unparametrized_model.eval()
            return unparametrized_model

        else:

            cls_instance.eval()
            return cls_instance

