

def get_execution_provider(config):
    match config["accelerator"]:
        case "cuda":
            return "CUDAExecutionProvider"
        case "cpu":
            return "CPUExecutionProvider"
        case _:
            raise Exception("Unsupported accelerator. Please set a supported accelerator in the config file.")
