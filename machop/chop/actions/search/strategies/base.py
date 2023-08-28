import torch
from chop.plt_wrapper import get_model_wrapper
from .forwards import ForwardMap
import logging


class StrategyBase:
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
    ):
        self.model_name = model_name
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
        self.config = config
        self.read_setup()

    def read_setup(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def feedback(self):
        raise NotImplementedError()

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

    def get_runner(self, mode):
        if mode in ["cls", "CLS"]:
            return self.cls_runner
        elif mode in ["lm", "LM"]:
            return self.lm_runner

    def cls_runner(self, model):
        i = 0
        loss, acc = 0, 0
        for batch in self.data_loader:
            metric = self.forward_map.forward(batch, model)
            loss += metric["loss"]
            acc += metric["acc"]
            i += 1
            if i >= self.num_batches:
                break
        avg_loss = loss / i
        avg_acc = acc / i
        return avg_loss, avg_acc

    def lm_runner(self, model):
        i = 0
        loss, perplexity = 0, 0
        for batch in self.data_loader:
            metric = self.forward_map.forward(batch, model)
            loss += metric["loss"]
            perplexity += metric["perplexity"]
            i += 1
            if i >= self.num_batches:
                break
        avg_loss = loss / i
        avg_perp = perplexity / i
        return avg_loss, avg_perp

    def run_mg(self, sample, search_space):
        mg = search_space.get_model(sample)
        metric = self.runner(mg.model)
        return mg, *metric

    def run_module_based_model(self, sample, search_space):
        breakpoint()
        pass
