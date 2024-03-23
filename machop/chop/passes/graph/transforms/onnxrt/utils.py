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