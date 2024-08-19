import torch
from torch.distributed.tensor.api import DTensorSpec, TensorMeta
from torch.distributed import DeviceMesh
from copy import deepcopy


from chop.tools import get_logger
from chop.distributed.tensor import DTensor


logger = get_logger(__name__)
logger.setLevel("INFO")


def rlog(msg):
    rank = torch.distributed.get_rank()
    if rank == 0:
        logger.info(msg)


class DTensorCache:
    _dtensor_dict: dict = {}

    def __init__(self):
        """
        This cache is needed to avoid expensive calls to _make_wrapper_subclass
        at runtime when wrapping local Tensor results in DTensor objects.
        """
        pass


def _create_dtensor(
    local_tensor,
    node_name,
    node_meta,
    result_name,
    torch_mesh,
):
    cached_name = f"{node_name}_{result_name}"
    cached_dtensor = DTensorCache._dtensor_dict.get(cached_name, None)

    # The DTensor is not found in the cache the first time each FX node is called
    if cached_dtensor is None:
        result_meta = node_meta["common"]["results"][result_name]

        dtensor = DTensor(
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

        DTensorCache._dtensor_dict[cached_name] = dtensor

        return dtensor

    # If the DTensor is found in the cache, replace the local tensor
    else:
        # Replace local tensor without constructing a new dtensor
        cached_dtensor._local_tensor = local_tensor

        # if DEBUG_MODE:
        #     assert cached dtensor has the same meta
        #     assert cached_dtensor._spec.placements == result_meta["dtensor_spec"].placements
        #     assert cached_dtensor._spec.tensor_meta.shape == result_meta["value"].shape
        #     assert cached_dtensor._spec.tensor_meta.stride == result_meta["value"].stride()
        #     assert cached_dtensor._spec.tensor_meta.dtype == local_tensor.dtype

        return cached_dtensor


def create_wrapper(node):

    target_fn = deepcopy(node.target)

    # todo: generalize
    torch_mesh = DeviceMesh(
        "cuda",
        mesh=torch.Tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
    )

    result_names = list(node.meta["mase"]["common"]["results"].keys())

    def dtensor_wrapper_fn(*args, **kwargs):
        out = target_fn(*args, **kwargs)

        if isinstance(out, (tuple, list)):
            outs = []
            for r_idx, r in enumerate(out):
                # if isinstance(r, DTensor):
                #     outs.append(r)
                if isinstance(r, torch.Tensor):
                    outs.append(
                        _create_dtensor(
                            local_tensor=r,
                            node_name=node.name,
                            node_meta=node.meta["mase"],
                            result_name=result_names[r_idx],
                            torch_mesh=torch_mesh,
                        )
                    )
                else:
                    outs.append(r)

            wrapped_out = tuple(outs)

        # In the event the OpDispatcher already wrapped a DTensor around
        # the local result, avoid reaching recursive depth limit
        # elif isinstance(out, DTensor):
        #     wrapped_out = out

        elif isinstance(out, torch.Tensor):
            wrapped_out = _create_dtensor(
                local_tensor=out,
                node_name=node.name,
                node_meta=node.meta["mase"],
                result_name=result_names[0],
                torch_mesh=torch_mesh,
            )

        else:
            wrapped_out = out

        return wrapped_out

    return dtensor_wrapper_fn


def insert_dtensor_wrapper_transform_pass(mg, pass_args={}):

    rlog("Inserting DTensor wrappers for call_function nodes")

    for node in mg.nodes:
        if node.op == "call_function":

            logger.debug(f"Inserting DTensor wrapper for {node.name}")
            node.target = create_wrapper(node)

        else:
            logger.debug(f"Skipping node {node.name} because it is not a call_function")

    return mg, {}
