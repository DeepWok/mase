import inspect
from typing import Literal
from ..models.utils import ModelSource


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
    }


def get_cf_args(model_info, task: str, model):
    """Get concrete forward args for freezing dynamic control flow in forward pass"""
    all_forward_kwargs = _get_default_args(model.forward)
    cf_args = {}
    if model_info.model_source == ModelSource.PATCHED:
        cf_args = model.patched_nodes["concrete_forward_args"]
    elif model_info.is_vision_model:
        cf_args = {}
    elif model_info.is_nlp_model:
        match task:
            case "classification" | "cls":
                required_input_args = ["input_ids", "attention_mask", "labels"]
            case "language_modeling" | "lm":
                required_input_args = ["input_ids", "attention_mask", "labels"]
            case "translation" | "tran":
                required_input_args = [
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
        for required_input_arg in required_input_args:
            all_forward_kwargs.pop(required_input_arg)
        cf_args = all_forward_kwargs
    else:
        raise RuntimeError(f"Unsupported model+task: {model_info.name}+{task}")
    return cf_args


def get_dummy_input(
    model_info,
    data_module,
    task: str,
) -> dict:
    """Create a single dummy input for a model. The dummy input is a single sample from the training set.

    Args:
        datamodule (MaseDataModule): a LightningDataModule instance (see machop/chop/dataset/__init__.py). Make sure the datamodule is prepared and setup.

        task (str): task name, one of ["cls", "classification", "lm", "language_modeling", "translation", "tran"]

        is_nlp_model (bool, optional): Whether the task is NLP task or not. Defaults to False.

    Returns:
        dict: a dummy input dict which can be passed to the wrapped lightning model's forward method, like model(**dummy_input)
    """
    assert (
        data_module.train_dataset is not None
    ), "DataModule is not setup. Please call data_module.prepare_data() and .setup()."
    index: int = 0
    device = "meta"
    train_iter = iter(data_module.train_dataloader())
    n_batches = len(data_module.train_dataloader())
    if index >= n_batches * data_module.batch_size:
        raise ValueError(f"index {index} is out of range.")
    batch_index = index // data_module.batch_size
    sample_index = index % data_module.batch_size
    for _ in range(batch_index):
        next(train_iter)

    if model_info.is_vision_model:
        match task:
            case "classification" | "cls":
                x, y = next(train_iter)
                x = x[[0], ...].to(device)
                dummy_inputs = {"x": x}
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
    elif model_info.is_nlp_model:
        match task:
            case "classification" | "cls":
                input_dict = next(train_iter)
                input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
                attention_mask = input_dict["attention_mask"][[sample_index], ...].to(
                    device
                )
                token_type_ids = input_dict["token_type_ids"][[sample_index], ...].to(
                    device
                )
                labels = input_dict["labels"][[sample_index], ...].to(device)
                dummy_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "labels": labels,
                }
            case "language_modeling" | "lm":
                input_dict = next(train_iter)
                input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
                attention_mask = input_dict["attention_mask"][[sample_index], ...].to(
                    device
                )
                labels = input_dict["labels"][[sample_index], ...].to(device)
                dummy_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            case "translation" | "tran":
                input_dict = next(train_iter)
                input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
                attention_mask = input_dict["attention_mask"][[sample_index], ...].to(
                    device
                )
                decoder_input_ids = input_dict["decoder_input_ids"][
                    [sample_index], ...
                ].to(device)
                decoder_attention_mask = input_dict["decoder_attention_mask"][
                    [sample_index], ...
                ].to(device)
                dummy_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                }
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
    else:
        raise RuntimeError(f"Unsupported model+task: {model_info.name}+{task}")

    return dummy_inputs


class InputGenerator:
    def __init__(
        self,
        model_info,
        data_module,
        task: str,
        which_dataloader: Literal["train", "val", "test"],
        max_batches: int = None,
    ) -> None:
        """
        Input generator for feeding batches to models. This is used for software passes.

        Args:
            datamodule (MyDataModule): a MyDataModule instance (see machop/chop/dataset/data_module.py). Make sure the datamodule is prepared and setup.
            max_batches (int, optional): Maximum number of batches to generate. Defaults to None will stop when reaching the last batch in dataloader.

        Returns:
            (dict): a dummy input dict which can be passed to the wrapped lightning model's forward method, like model(**dummy_input)
        """
        assert (
            getattr(data_module, f"{which_dataloader}_dataset") is not None
        ), "DataModule is not setup. Please call data_module.prepare_data() and .setup()."
        self.model_info = model_info
        self.task = task

        self.batch_size = data_module.batch_size
        self.dataloader = getattr(data_module, f"{which_dataloader}_dataloader")()
        self.dataloader_iter = iter(self.dataloader)

        self.max_batches = max_batches
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self.max_batches is not None and self.current_batch >= self.max_batches:
            raise StopIteration

        if self.model_info.is_vision_model:
            match self.task:
                case "classification" | "cls":
                    x, y = next(self.dataloader_iter)
                    inputs = {"x": x}
                case _:
                    raise ValueError(
                        f"Task {self.task} is not supported for {self.model_info.name}"
                    )
        elif self.model_info.is_nlp_model:
            match self.task:
                case "classification" | "cls":
                    input_dict = next(self.dataloader_iter)
                    inputs = {
                        "input_ids": input_dict["input_ids"],
                        "attention_mask": input_dict["attention_mask"],
                        "labels": input_dict["labels"],
                    }
                    if "token_type_ids" in input_dict:
                        inputs["token_type_ids"] = input_dict["token_type_ids"]
                case "language_modeling" | "lm":
                    input_dict = next(self.dataloader_iter)
                    inputs = {
                        "input_ids": input_dict["input_ids"],
                        "attention_mask": input_dict["attention_mask"],
                        "labels": input_dict["labels"],
                    }
                case "translation" | "tran":
                    input_dict = next(self.dataloader_iter)
                    inputs = {
                        "input_ids": input_dict["input_ids"],
                        "attention_mask": input_dict["attention_mask"],
                        "decoder_input_ids": input_dict["decoder_input_ids"],
                        "decoder_attention_mask": input_dict["decoder_attention_mask"],
                    }
                case _:
                    raise ValueError(
                        f"Task {self.task} is not supported for {self.model_info.name}"
                    )
        else:
            raise RuntimeError(
                f"Unsupported model+task: {self.model_info.name}+{self.task}"
            )

        self.current_batch += 1
        return inputs
