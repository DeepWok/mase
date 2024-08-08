import torch
from torch.distributed._tensor.api import DTensorSpec, TensorMeta
from torch.distributed import DeviceMesh
from copy import deepcopy


from chop.tools import get_logger
from chop.distributed.tensor import DTensor


logger = get_logger(__name__)
logger.setLevel("DEBUG")


def _create_dtensor(local_tensor, result_meta, torch_mesh):
    return DTensor(
        local_tensor=local_tensor,
        spec=DTensorSpec(
            mesh=torch_mesh,
            placements=result_meta["dtensor_spec"].placements,
            tensor_meta=TensorMeta(
                shape=result_meta["value"].shape,
                stride=result_meta["value"].stride(),
                dtype=local_tensor.dtype,
            ),
        ),
        requires_grad=local_tensor.requires_grad,
    )


def create_wrapper(node):

    target_fn = deepcopy(node.target)

    torch_mesh = DeviceMesh(
        "cuda",
        mesh=torch.Tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
    )

    result_names = list(node.meta["mase"]["common"]["results"].keys())

    def dtensor_wrapper_fn(*args, **kwargs):
        out = target_fn(*args, **kwargs)

        # In the event the OpDispatcher already wrapped a DTensor around
        # the local result, avoid reaching recursive depth limit
        if isinstance(out, (tuple, list)):
            outs = []
            for r_idx, r in enumerate(out):
                if isinstance(r, DTensor):
                    outs.append(r)
                elif isinstance(r, torch.Tensor):
                    outs.append(
                        _create_dtensor(
                            r,
                            result_meta=node.meta["mase"]["common"]["results"][
                                result_names[r_idx]
                            ],
                            torch_mesh=torch_mesh,
                        )
                    )
                else:
                    outs.append(r)

            wrapped_out = tuple(outs)

        elif isinstance(out, DTensor):
            wrapped_out = out

        elif isinstance(out, torch.Tensor):
            wrapped_out = _create_dtensor(
                out,
                result_meta=node.meta["mase"]["common"]["results"][result_names[0]],
                torch_mesh=torch_mesh,
            )

        else:
            wrapped_out = out

        return wrapped_out

    return dtensor_wrapper_fn


def insert_dtensor_wrapper_transform_pass(mg, pass_args={}):

    logger.info("Inserting DTensor wrappers for call_function nodes")

    for node in mg.nodes:
        if node.op == "call_function":

            logger.info(f"Inserting DTensor wrapper for {node.name}")
            node.target = create_wrapper(node)

        else:
            logger.warning(
                f"Skipping node {node.name} because it is not a call_function"
            )

    return mg, {}
