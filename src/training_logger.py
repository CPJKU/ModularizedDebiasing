from pathlib import Path
from typing import Union, Optional, Dict, List
from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    delta: float = 1e-8

    @staticmethod
    def suffix_fn(suffix):
         return "" if len(suffix)==0 else f"_{suffix}"

    def __init__(
        self,
        log_dir: Union[str, Path],
        logger_name: str,
        logging_step: int
    ):
        assert logging_step > 0, "logging_step needs to be > 0"

        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)

        self.log_dir = log_dir
        self.logger_name = logger_name
        self.logging_step = logging_step

        self.writer = SummaryWriter(log_dir / logger_name)

        self.reset()

    def validation_loss(self, eval_step: int, result: dict, suffix: str = ''):
        suffix = self.suffix_fn(suffix)
        for name, value in sorted(result.items(), key=lambda x: x[0]):
            self.writer.add_scalar(f'val/{name}{suffix}', value, eval_step)
    
    def test_loss(self, eval_step: int, result: dict, suffix: str = ''):
        suffix = self.suffix_fn(suffix)
        for name, value in sorted(result.items(), key=lambda x: x[0]):
            self.writer.add_scalar(f'test/{name}{suffix}', value, eval_step)

    def step_loss(self, step: int, loss: Union[float, Dict[str, float]], lr: Optional[float] = None, increment_steps: bool = True, suffix: str = ''):
        self.steps += int(increment_steps)

        if not isinstance(loss, dict):
            loss = {suffix: loss}

        for k,v in loss.items():
            try:
                self.loss_dict[k] += v
            except KeyError:
                self.loss_dict[k] = v

        if (self.steps > 0) and (self.steps % self.logging_step == 0):
            logs = {f"step_loss{self.suffix_fn(k)}": v / self.steps for k,v in self.loss_dict.items()}
            if lr:
                logs["step_learning_rate"] = lr
            for key, value in logs.items():
                self.writer.add_scalar(f'train/{key}', value, step // self.logging_step)

            self.steps = 0
            self.loss_dict = {}

    def non_zero_params(self, step, n_p, n_p_zero, n_p_between, suffix: str = ''):
        suffix = self.suffix_fn(suffix)
        d = {
            "zero_ratio": n_p_zero / n_p,
            "between_ratio": n_p_between / n_p
        }
        for k,v in d.items():
            self.writer.add_scalar(f"train/{k}{suffix}", v, step)

    def is_best(self, val: float, ascending: bool, id: str = "loss") -> bool:
        try:
            best_val = self.best_eval_metric[id]
        except KeyError:
            self.best_eval_metric[id] = val
            return True

        if ascending:
            check = val < (best_val + self.delta)
        else:
            check = val > (best_val - self.delta)
        if check:
            self.best_eval_metric[id] = val
        return check     

    def reset(self):
        self.steps = 0
        self.loss_dict = {}
        self.best_eval_metric = {}