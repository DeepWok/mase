import inspect

from chop.models import (
    nlp_models,
    patched_model_cls_to_required_input_args,
    patched_nlp_models,
    vision_models,
)


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
    }


def get_cf_args(model_name: str, task: str, model):
    """Get concrete forward args for freezing dynamic control flow in forward pass"""
    default_forward_kwargs = _get_default_args(model.forward)
    cf_args = {}
    if (
        model_name in patched_nlp_models
        and type(model) in patched_model_cls_to_required_input_args
    ):
        required_input_args = patched_model_cls_to_required_input_args[type(model)]
        for required_input_arg in required_input_args:
            default_forward_kwargs.pop(required_input_arg)
        cf_args = default_forward_kwargs
    elif model_name in patched_nlp_models or model_name in nlp_models:
        if task in ["cls", "classification"]:
            required_input_args = ["input_ids", "attention_mask"]
        elif task in ["lm", "language_modeling"]:
            required_input_args = ["input_ids", "attention_mask", "labels"]
            # required_input_args = ["input_ids", "attention_mask"]
        else:
            # translation
            required_input_args = [
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
        for required_input_arg in required_input_args:
            default_forward_kwargs.pop(required_input_arg)
        cf_args = default_forward_kwargs
    elif model_name in vision_models:
        # Currently the only input to vision model is a Tensor x
        cf_args = {}
    else:
        raise RuntimeError(f"Unsupported model+task: {model_name}+{task}")
    return cf_args


def get_dummy_input(
    datamodule,
    task: str,
    is_nlp_model: bool = False,
) -> dict:
    """Create a dummy input for a model. The dummy input is a single sample from the training set.

    Args:
        datamodule (MyDataModule): a MyDataModule instance (see machop/chop/dataset/data_module.py). Make sure the datamodule is prepared and setup.

        task (str): task name, one of ["cls", "classification", "lm", "language_modeling", "translation", "tran"]

        is_nlp_model (bool, optional): Whether the task is NLP task or not. Defaults to False.

    Returns:
        dict: a dummy input dict which can be passed to the wrapped lightning model's forward method, like model(**dummy_input)
    """
    index: int = 0
    device = "meta"
    train_iter = iter(datamodule.train_dataloader())
    n_batches = len(datamodule.train_dataloader())
    if index >= n_batches * datamodule.batch_size:
        raise ValueError(f"index {index} is out of range.")
    batch_index = index // datamodule.batch_size
    sample_index = index % datamodule.batch_size
    for _ in range(batch_index):
        next(train_iter)

    if task in ["cls", "classification"] and not is_nlp_model:
        x, y = next(train_iter)
        x = x[[0], ...].to(device)
        dummy_inputs = {"x": x}
    elif task in ["cls", "classification"] and is_nlp_model:
        input_dict = next(train_iter)
        input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
        attention_mask = input_dict["attention_mask"][[sample_index], ...].to(device)
        token_type_ids = input_dict["token_type_ids"][[sample_index], ...].to(device)
        labels = input_dict["labels"][[sample_index], ...].to(device)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
    elif task in ["lm", "language_modeling"]:
        input_dict = next(train_iter)
        input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
        attention_mask = input_dict["attention_mask"][[sample_index], ...].to(device)
        labels = input_dict["labels"][[sample_index], ...].to(device)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    elif task in ["translation", "tran"]:
        input_dict = next(train_iter)
        input_ids = input_dict["input_ids"][[sample_index], ...].to(device)
        attention_mask = input_dict["attention_mask"][[sample_index], ...].to(device)
        decoder_input_ids = input_dict["decoder_input_ids"][[sample_index], ...].to(
            device
        )
        decoder_attention_mask = input_dict["decoder_attention_mask"][
            [sample_index], ...
        ].to(device)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    return dummy_inputs
