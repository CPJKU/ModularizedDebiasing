import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Union, Callable, Dict, Optional

from src.models.model_heads import ClfHead
from src.models.model_base import BaseModel
from src.training_logger import TrainLogger
from src.utils import dict_to_device, evaluate_model
from transformers.adapters import PfeifferConfig


class TaskAdapter(BaseModel):

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = .3,
        n_hidden: int = 0,
        rf: int = 16,
        bottleneck: bool = False,
        bottleneck_dim: Optional[int] = None,
        bottleneck_dropout: Optional[float] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        self.num_labels = num_labels
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.has_bottleneck = bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_dropout = bottleneck_dropout
        self.rf = rf

        adap_config = PfeifferConfig(reduction_factor=self.rf)
        self.encoder.add_adapter("task",adap_config)
        self.encoder.train_adapter("task",adap_config)

        # bottleneck layer
        if self.has_bottleneck:
            self.bottleneck = ClfHead(self.hidden_size, bottleneck_dim, dropout=bottleneck_dropout)
            self.in_size_heads = bottleneck_dim
        else:
            self.bottleneck = torch.nn.Identity()
            self.in_size_heads = self.hidden_size

        self.task_head = ClfHead([self.in_size_heads]*(n_hidden+1), num_labels, dropout=dropout)

    def _forward(self, **x) -> torch.Tensor:
        hidden = super()._forward(**x)
        return self.bottleneck(hidden)

    def forward(self, **x) -> torch.Tensor:
        return self.task_head(self._forward(**x))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs: int,
        learning_rate: float,
        learning_rate_bottleneck: float,
        learning_rate_head: float,
        optimizer_warmup_steps: int,
        max_grad_norm: float,
        output_dir: Union[str, os.PathLike],
        cooldown: int,
        checkpoint_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:

        self.global_step = 0

        train_steps = len(train_loader) * num_epochs
        self._init_optimizer_and_schedule(
            train_steps,
            learning_rate,
            learning_rate_head,
            learning_rate_bottleneck,
            optimizer_warmup_steps
        )

        self.zero_grad()

        train_str = "Epoch {}, {}"
        str_suffix = lambda d: ", ".join([f"{k}: {v}" for k,v in d.items()])

        performance_decrease_counter = 0
        train_iterator = trange(num_epochs, desc=train_str.format(0, ""), leave=False, position=0)
        for epoch in train_iterator:

            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm
            )

            result = self.evaluate(
                val_loader,
                loss_fn,
                pred_fn,
                metrics
            )

            logger.validation_loss(epoch, result, "task")

            train_iterator.set_description(
                train_str.format(epoch, str_suffix(result)), refresh=True
            )

            if logger.is_best(result["loss"], ascending=True):
                cpt = self.save_checkpoint(Path(output_dir), checkpoint_name, seed)
                cpt_result = result
                cpt_epoch = epoch
                performance_decrease_counter = 0
            else:
                performance_decrease_counter += 1

            if performance_decrease_counter>cooldown:
                break

        print("Final result after " + train_str.format(epoch, str_suffix(result)))
        print("Best result: " + train_str.format(cpt_epoch, str_suffix(cpt_result)))

        return cpt


    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        pred_fn: Callable,
        metrics: Dict[str, Callable]
    ) -> dict:
        self.eval()

        forward_fn = lambda x: self(**x)

        return evaluate_model(
            self,
            val_loader,
            loss_fn,
            pred_fn,
            metrics,
            forward_fn=forward_fn
        )

    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        max_grad_norm: float
    ) -> None:
        self.train()

        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):

            inputs, labels = batch[0], batch[1]
            inputs = dict_to_device(inputs, self.device)
            outputs = self(**inputs)
            loss = loss_fn(outputs, labels.to(self.device))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)

            self.optimizer.step()
            # self.scheduler.step()
            self.zero_grad()

            logger.step_loss(self.global_step, {"total": loss.item(), "task": loss.item()})

            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)

            self.global_step += 1


    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        learning_rate_head: float,
        learning_rate_bottleneck: float = 1e-4,
        num_warmup_steps: int = 0
    ) -> None:
        optimizer_params = [
            {
                "params": self.encoder.parameters(),
                "lr": learning_rate
            },
            {
                "params": self.bottleneck.parameters(),
                "lr": learning_rate_bottleneck
            },
            {
                "params": self.task_head.parameters(),
                "lr": learning_rate_head
            }
        ]
        self.optimizer = AdamW(optimizer_params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        # )


    def make_checkpoint_name(
        self,
        seed: Optional[int] = None
    ):
        filename_parts = [
            self.model_name.split('/')[-1],
            "task_baseline",
            "cp_init" if self.state_dict_init else None,
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
            "num_labels": self.num_labels,
            "dropout": self.dropout,
            "n_hidden": self.n_hidden,
            "reduction_factor": self.rf,
            "bottleneck": self.has_bottleneck,
            "bottleneck_dim": self.bottleneck_dim,
            "bottleneck_dropout": self.bottleneck_dropout,
            "encoder_state_dict": self.encoder_module.state_dict(),
            "bottleneck_state_dict": self.bottleneck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict()
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_name is None:
            checkpoint_name = self.make_checkpoint_name(seed)
        filepath = output_dir / checkpoint_name
        torch.save(info_dict, filepath)
        self.encoder_module.save_adapter(output_dir/'Adp_T', "task")
        return filepath


    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, os.PathLike],
        map_location: Union[str, torch.device] = torch.device('cpu')
    ) -> torch.nn.Module:
        info_dict = torch.load(filepath, map_location=map_location)

        cls_instance = cls(
            info_dict['model_name'],
            info_dict['num_labels'],
            info_dict['dropout'],
            info_dict['n_hidden'],
            info_dict['reduction_factor'],
            info_dict['bottleneck'],
            info_dict['bottleneck_dim'],
            info_dict['bottleneck_dropout']
        )
        cls_instance.encoder.load_state_dict(info_dict['encoder_state_dict'])
        cls_instance.bottleneck.load_state_dict(info_dict['bottleneck_state_dict'])
        cls_instance.task_head.load_state_dict(info_dict['task_head_state_dict'])

        cls_instance.eval()

        return cls_instance
