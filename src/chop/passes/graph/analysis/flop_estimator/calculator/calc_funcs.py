import torch


def calculate_funcs(function, fn_args, fn_kwargs, out_data):
    # Collect computation statistics.
    if function.__name__ == "add":
        # One computation per input pixel - window size is chosen adaptively
        # and windows never overlap (?).
        if len(fn_args) > 1:
            input_size = fn_args[0].numel()
            output_size = out_data[0].numel()
            computations = input_size
            backward_computations = input_size
        else:
            raise ValueError(
                f"Unsupported number of arguments for function {function.__name__}"
            )
        return {
            "total_parameters": 0,
            "computations": computations,
            "backward_computations": backward_computations,
            "input_buffer_size": input_size,
            "output_buffer_size": output_size,
        }
    else:
        print("Unsupported function type for analysis:", function.__name__)
