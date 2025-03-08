import torch
import time

def measure_inference_speed(model, sample_batch, device='cuda', num_warmup=5, num_runs=20):
    """
    Measures average inference time (seconds) for `num_runs` forward passes.
    Assumes `sample_batch` is already on CPU and will be moved to `device`.
    """
    model.to(device)
    model.eval()

    # Convert the sample batch tensors to device
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            sample_batch[k] = v.to(device)

    # Warm-up runs
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(**sample_batch)

    torch.cuda.synchronize()
    start_time = time.time()

    # Timed runs
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**sample_batch)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time
