import logging
import os
import torch

from chop.plt_wrapper import get_model_wrapper
from .forwards import ForwardMap

"""
Search runner manages the following:
1. data used for the search
2. the quality metric produced
"""


class SearchRunnerBase:
    def __init__(
        self,
        model_name,
        model,
        mg,
        task,
        info,
        data_module,
        accelerator,
        config,
        save_dir,
    ) -> None:
        self.model = model
        self.mg = mg

        self.info = info
        self.task = task
        self.config = config

        self._prepare_loader(data_module)
        self._set_accelerator(accelerator)
        self.save_dir = save_dir
        self._create_logger()
        self.training = config.get("training")
        self.model_wrapper = get_model_wrapper(model_name, task)

        self.forward_map = ForwardMap(task, info)

    def _create_logger(self):
        logger = logging.getLogger("search")
        logger.setLevel(logging.INFO)
        self.logger = logger

    def _prepare_loader(self, data_module):
        data_module.prepare_data()
        data_module.setup()
        self.data_loader = getattr(data_module, self.config["data_loader"])()
        self.num_batches = self.config["num_batches"]

    def _set_accelerator(self, accelerator):
        if accelerator == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif accelerator == "gpu":
            self.device = torch.device("cuda:0")
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise RuntimeError(f"Unsupported accelerator {accelerator}")

    def runner(self, model):
        i = 0
        loss = 0
        for batch in self.data_loader:
            metric = self.forward_map.forward(batch, model)
            loss += metric["loss"]
            i += 1
            if i >= self.num_batches:
                break
        avg_loss = loss / i
        return avg_loss
