from torch.utils.data import DataLoader, Subset
import logging


def get_execution_provider(accelerator):
    logger = logging.getLogger(__name__)
    match accelerator:
        case "cuda":
            logger.info("Using CUDA as ONNX execution provider.")
            return "CUDAExecutionProvider"
        case "cpu":
            logger.info("Using CPU as ONNX execution provider.")
            return "CPUExecutionProvider"
        case _:
            raise Exception(
                "Unsupported accelerator for ONNX execution provider. Please set a supported accelerator in the config file."
            )


def get_calibrator_dataloader(original_dataloader, num_batches=200):
    # Get the batch size from the original DataLoader
    batch_size = original_dataloader.batch_size

    # Calculate the number of samples needed for the desired number of batches
    num_samples = num_batches * batch_size

    # Assuming the dataset is accessible through the DataLoader
    original_dataset = original_dataloader.dataset

    # Create a subset of the original dataset
    # Note: This assumes that indexing the dataset returns individual samples.
    # If your dataset returns batches, this approach needs to be adjusted.
    subset_dataset = Subset(original_dataset, range(num_samples))

    # Create a new DataLoader from the subset dataset
    calibrator_dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,  # Typically calibration data isn't shuffled, adjust as needed.
        num_workers=original_dataloader.num_workers,
        pin_memory=original_dataloader.pin_memory,
    )

    return calibrator_dataloader


import torch


def convert_dataloader_to_onnx_dataset_dict(data_loader, input_names, device="cpu"):
    """
    Converts batches from a DataLoader to feed dictionaries suitable for ONNX model input.

    Parameters:
        data_loader: DataLoader yielding batches of data.
        input_names: List of strings, names of the input tensors required by the ONNX model.
        device: The device type ('cpu' or 'cuda') for the DataLoader tensors.

    Returns:
        List of dictionaries, each mapping from ONNX model input names to batch data.
    """
    feed_dicts = []  # This will hold the feed dictionaries for all batches
    for batch in data_loader:
        # Convert batch data to numpy arrays and create feed_dict
        # Check if your batch is a single tensor or a tuple/list of tensors:
        if isinstance(batch, (list, tuple)):
            # Assuming the order of tensors in batch corresponds to the order of input names
            feed_dict = {
                name: tensor.to(device).numpy()
                for name, tensor in zip(input_names, batch)
            }
        else:
            # Single tensor assumed, corresponding to the first input name
            feed_dict = {input_names[0]: batch.to(device).numpy()}

        feed_dicts.append(feed_dict)

    return feed_dicts
