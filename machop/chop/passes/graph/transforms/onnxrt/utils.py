from torch.utils.data import DataLoader, Subset

def get_execution_provider(config):
    match config["accelerator"]:
        case "cuda":
            return "CUDAExecutionProvider"
        case "cpu":
            return "CPUExecutionProvider"
        case _:
            raise Exception("Unsupported accelerator. Please set a supported accelerator in the config file.")


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
        pin_memory=original_dataloader.pin_memory
    )
    
    return calibrator_dataloader

def convert_dataloader_to_numpy(dataloader, input_names=None):
    """
    Converts dataloader to a list of dictionaries with NumPy arrays.
    If input_names is provided, it uses it to map tensors to the expected ONNX input names.
    """
    numpy_data = []
    for batch in dataloader:
        # Convert batch to numpy arrays
        if isinstance(batch, (list, tuple)):
            batch_data = [item.numpy() for item in batch]
        elif isinstance(batch, dict):
            batch_data = {key: value.numpy() for key, value in batch.items()}
        else:  # Assuming the batch itself is a tensor
            batch_data = batch.numpy()
        
        # Map to input names if provided
        if input_names:
            assert isinstance(batch_data, list) and len(batch_data) == len(input_names)
            batch_data = {name: array for name, array in zip(input_names, batch_data)}
        
        numpy_data.append(batch_data)
    return numpy_data