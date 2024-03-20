

def get_execution_provider(config):
    EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return (
        "CUDAExecutionProvider"
        if config["accelerator"] == "cuda"
        else "CPUExecutionProvider"
    )