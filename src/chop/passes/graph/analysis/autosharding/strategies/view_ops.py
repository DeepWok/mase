# Adapted from Pytorch Distributed DTensor API.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/ops/view_ops.py

from dataclasses import dataclass
from typing import (
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch.distributed._tensor._op_schema import (
    OpStrategy,
    PlacementStrategy,
)
from torch.distributed._tensor.api import Shard
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    normalize_dim,
    normalize_dims,
    prod,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Placement,
    Replicate,
    TensorMeta,
)

Shape = Tuple[int, ...]


@dataclass
class DimSpec:
    """Specifies how an output dimension maps to an input dimension."""

    def inputs(self) -> Iterable["DimSpec"]:
        return ()


# Rules that map each dimension of the output to dimensions of the input tensor
DimMap = Tuple[DimSpec, ...]


@dataclass
class Singleton(DimSpec):
    """Output dimension is a singleton."""

    pass


@dataclass
class InputDim(DimSpec):
    """Output dimension maps directly to an input dimension."""

    input_dim: int


@dataclass
class Broadcast(DimSpec):
    """Output is the broadcast of a singleton input dimension."""

    dim: DimSpec
    dim_size: int

    @classmethod
    def new(cls, dim: DimSpec, dim_size: int) -> DimSpec:
        return Broadcast(dim, dim_size)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.dim,)


@dataclass
class NewDim(DimSpec):
    """This is a new dimension created by the op."""

    size: int

    @classmethod
    def new(cls, size: int) -> DimSpec:
        return Singleton() if size == 1 else NewDim(size)


@dataclass
class Repeat(DimSpec):
    """Output dimension is the input dimension repeated n-times."""

    input_dim: DimSpec
    times: int

    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec:
        if times == 1:
            return dim
        elif isinstance(dim, Singleton):
            # repeating a singleton is the same as broadcasting it
            return Broadcast(dim, times)
        else:
            return Repeat(dim, times)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


@dataclass
class Flatten(DimSpec):
    """Flatten a set of input dimensions, ensuring right-most adjacent elements remain adjacent in the output."""

    input_dims: Sequence[DimSpec]

    @classmethod
    def new(cls, dims: Sequence[DimSpec]) -> DimSpec:
        if len(dims) == 0:
            # flattening a scalar leads to a singleton
            return Singleton()
        elif len(dims) == 1:
            # flattening a single dimension is no-op
            return dims[0]
        else:
            return Flatten(dims)

    def inputs(self) -> Iterable[DimSpec]:
        return self.input_dims


@dataclass
class Split(DimSpec):
    """
    This dimension is a member of a decomposition of the input dim.

    Note that input_dim itself could be a Flattened set of input dims.
    """

    input_dim: DimSpec
    group_shape: Shape
    split_id: int

    @classmethod
    def new(cls, dim: DimSpec, group_shape: Tuple[int, ...], idx: int) -> DimSpec:
        assert len(group_shape) > 0
        if len(group_shape) == 1:
            # not really a group, just return the input dim back
            assert idx == 0
            return dim
        elif group_shape[idx] == 1:
            return Singleton()
        else:
            # remove singletons from group
            # group_mapping = [(new_index, (shape, old_index)) ...]
            group_mapping = list(
                enumerate((s, i) for i, s in enumerate(group_shape) if s != 1)
            )
            new_group_shape = tuple(m[1][0] for m in group_mapping)
            new_idx = next(filter(lambda x: x[1][1] == idx, group_mapping))[0]
            return Split(dim, new_group_shape, new_idx)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


def dim_pad_left(ndim: int, min_dims: int) -> DimMap:
    return (Singleton(),) * max(0, min_dims - ndim) + tuple(
        InputDim(i) for i in range(ndim)
    )


def dim_atleast_3d(ndim: int) -> DimMap:
    if ndim == 0:
        return (Singleton(), Singleton(), Singleton())
    elif ndim == 1:
        return (Singleton(), InputDim(0), Singleton())
    elif ndim == 2:
        return (InputDim(0), InputDim(1), Singleton())
    else:
        return tuple(InputDim(i) for i in range(ndim))


def expand(input_shape: Shape, shape: Shape) -> DimMap:
    """Implement broadcast on multiple dimensions."""
    assert len(shape) >= len(input_shape)

    # 1. create padded input dimensions
    padded_input = dim_pad_left(len(input_shape), len(shape))
    # 2. check that input shapes are compatible
    mapping = []
    for p, desired_s in zip(padded_input, shape):
        if isinstance(p, Singleton):
            actual_s = 1
            assert desired_s >= 0
        else:
            assert isinstance(p, InputDim), f"DimSpec not supported in expand: {p}"
            actual_s = input_shape[p.input_dim]
            assert actual_s == 1 or desired_s == -1 or desired_s == actual_s
        mapping.append(
            p
            if desired_s in (1, -1) or desired_s == actual_s
            else Broadcast.new(p, desired_s)
        )
    return tuple(mapping)


def normalize_sizes(sizes: Union[Shape, Tuple[Shape]]) -> Shape:
    if isinstance(sizes[0], int):
        return cast(Shape, sizes)
    elif len(sizes) == 1:
        return cast(Shape, sizes[0])  # type: ignore[redundant-cast]
    else:
        raise RuntimeError("Size must be int... or tuple")


def dim_flatten(ndim: int, start_dim=0, end_dim=-1) -> DimMap:
    if ndim == 0:
        return (Singleton(),)
    elif ndim == 1:
        return (InputDim(0),)
    else:
        # only flattening dims from start_dim to end_dim (inclusive)
        # other dims are passed through
        if end_dim < 0:
            end_dim += ndim
        results: List[DimSpec] = [InputDim(i) for i in range(start_dim)]
        results.append(
            Flatten.new(tuple(InputDim(i) for i in range(start_dim, end_dim + 1)))
        )
        results.extend([InputDim(i) for i in range(end_dim + 1, ndim)])
        return tuple(results)


def dim_movedim(
    ndim: int,
    input: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> DimMap:
    input = normalize_dims(input, ndim)
    destination = normalize_dims(destination, ndim)

    assert len(input) == len(destination)
    input_set = set(input)
    assert len(input_set) == len(input), "Found repeated input dims"
    assert len(set(destination)) == len(destination), "Found repeated output dims"
    assert max(input) < ndim
    assert max(destination) < ndim

    dest = [-1] * ndim
    for i, d in zip(input, destination):
        dest[d] = i

    unused_inputs_iter = iter(i for i in range(ndim) if i not in input_set)
    for i in range(ndim):
        if dest[i] == -1:
            dest[i] = next(unused_inputs_iter)

    return tuple(InputDim(i) for i in dest)


def dim_repeat(ndim: int, sizes: Shape) -> DimMap:
    sizes = normalize_sizes(sizes)
    assert (
        len(sizes) >= ndim
    ), f"Number of dimensions of repeat dims {sizes} can not be smaller than number of dimensions of tensor {ndim}."
    pad = len(sizes) - ndim
    return tuple(Repeat.new(Singleton(), s) for s in sizes[:pad]) + tuple(
        Repeat.new(InputDim(i), s) for i, s in enumerate(sizes[pad:])
    )


def infer_size(total_size: int, sizes: Shape) -> Shape:
    """
    One dimension input to view may be "-1".

    Infer the size of this dimension given the total_size.
    """
    infers = [i for i, s in enumerate(sizes) if s == -1]
    size = prod(sizes)
    assert len(infers) <= 1, "can only infer one size"
    if infers:
        size = -size
        missing_size = total_size // size
        assert (
            total_size % size == 0
        ), f"size inferred for -1 is not integral {sizes} should have {total_size} elements."
        return tuple(s if s != -1 else missing_size for s in sizes)
    assert size == total_size, f"sizes do not match {total_size} vs {size}"
    return sizes


def view_groups(from_size: Shape, to_size: Shape) -> DimMap:
    """
    Decompose a reshape operation into forwarding, flattening, or splitting dimensions for each output dimension.

    A view or reshape operation can be decomposed into a set of 3 types of smaller operations:
    1) Forward a dimension from input to output
    2) Flatten a set of dimensions into a single dimension
    3) Split one dimension into multiple dimensions

    view_groups identifies these operations and returns, for each output dimension, what
    is operation was performed in the input dimension. For example:

        view_groups([2, 3, 4], [2, 12]) -> (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2)))
        )

    - ouptut dimension 0 maps to input dimension 0
    - output dimension 1 maps to a flattened input dimensions 1 and 2


        view_groups([2, 3], [3, 2]) -> (
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
        )

    - in the above, input is flattened into a single dimension and then split
      into two separate dimensions with different sizes from the input.
    """
    from_nelem = prod(from_size)
    to_size = infer_size(from_nelem, normalize_sizes(to_size))

    assert from_nelem == prod(to_size), "Total view shape does not add up"

    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)

    result_pp = []

    while from_idx < from_len or to_idx < to_len:
        from_group_dim, to_group_shape = [], []

        if from_idx >= from_len:
            f = 1
        else:
            f = from_size[from_idx]
            from_group_dim.append(from_idx)
            from_idx += 1

        if to_idx >= to_len:
            t = 1
        else:
            t = to_size[to_idx]
            to_group_shape.append(t)
            to_idx += 1

        # if any of the groups is singleton, great, we need to backtrack though
        if f == 1 and t != 1:
            # produces ([1], [])
            to_idx -= 1
            to_group_shape = []
        elif f != 1 and t == 1:
            # produces ([], [1])
            from_idx -= 1
            from_group_dim = []
        else:
            # produces ([1], [1]),  ([2], [2]), ([2,3], [6])
            while f != t:
                if f < t:
                    nf = from_size[from_idx]
                    from_group_dim.append(from_idx)
                    from_idx += 1
                    f *= nf
                else:
                    nt = to_size[to_idx]
                    to_group_shape.append(nt)
                    to_idx += 1
                    t *= nt

        if len(to_group_shape) > 0:
            flattened = Flatten.new(
                tuple(InputDim(fi) for fi in from_group_dim if from_size[fi] > 1)
            )
            result_pp += [
                Split.new(flattened, tuple(to_group_shape), i)
                for i in range(len(to_group_shape))
            ]

    return tuple(result_pp)


def dim_tile(ndim: int, dims: Tuple[int, ...]) -> DimMap:
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    assert dim1 < ndim
    assert dim2 < ndim
    dimmap = [InputDim(i) for i in range(ndim)]
    swapdim = dimmap[dim1]
    dimmap[dim1] = dimmap[dim2]
    dimmap[dim2] = swapdim
    return tuple(dimmap)


def dim_squeeze(shape: Shape, dim: Optional[int] = None) -> DimMap:
    # FIXME: this is wrong when dim=None and one of the dimensions
    # equals size of the mesh. For example squeeze(DTensor(tensor(4), Shard[0])) could
    # end up as squeeze(tensor(1)) if we have 4 devices; this would lead to
    # removal of a dimension that is not actually a singleton.
    return tuple(
        InputDim(i)
        for i, s in enumerate(shape)
        if s > 1 or (dim is not None and i != normalize_dim(dim, len(shape)))
    )


def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    dims = tuple(InputDim(i) for i in range(ndim))
    if dim < 0:
        dim += ndim + 1
    return dims[:dim] + (Singleton(),) + dims[dim:]


def dim_view_as_real(shape: Shape) -> DimMap:
    ndim = len(shape)
    results: List[DimSpec] = [InputDim(i) for i in range(ndim - 1)]
    # each complex number is split into two real numbers,
    # resulting in one more dimension of size 2
    results.append(Split(InputDim(ndim - 1), (shape[-1], 2), 0))
    results.append(Split(InputDim(ndim - 1), (shape[-1], 2), 1))
    return tuple(results)


dim_maps: Dict[Callable[..., torch.Tensor], Callable[..., DimMap]] = {
    torch.atleast_1d: lambda x: dim_pad_left(x.ndim, 1),
    torch.atleast_2d: lambda x: dim_pad_left(x.ndim, 2),
    torch.atleast_3d: lambda x: dim_atleast_3d(x.ndim),
    torch.broadcast_to: lambda input, shape: expand(input.shape, shape),
    torch.flatten: lambda tensor: dim_flatten(tensor.ndim),
    torch.movedim: lambda input, source, destination: dim_movedim(
        input.ndim, source, destination
    ),
    torch.permute: lambda input, *dims: tuple(
        InputDim(i) for i in normalize_dims(tuple(dims), input.ndim)
    ),
    torch.ravel: lambda tensor: dim_flatten(tensor.ndim),
    torch.reshape: lambda input, shape: view_groups(input.shape, shape),
    torch.squeeze: lambda input, dim=None: dim_squeeze(input.shape, dim),
    torch.tile: lambda input, dims: dim_tile(input.ndim, dims),
    torch.transpose: lambda input, dim0, dim1: dim_transpose(input.ndim, dim0, dim1),
    torch.unsqueeze: lambda input, dim: dim_unsqueeze(input.ndim, dim),
    torch.view_as_complex: lambda input: dim_flatten(input.ndim, input.ndim - 2),
    torch.view_as_real: lambda input: dim_view_as_real(input.shape),
    Tensor.expand: lambda self, *sizes: expand(self.shape, normalize_sizes(sizes)),
    Tensor.repeat: lambda self, *sizes: dim_repeat(self.ndim, sizes),
    Tensor.reshape: lambda self, *shape: view_groups(self.shape, shape),
    Tensor.view: lambda input, *shape: view_groups(input.shape, shape),
    # here
    Tensor.permute: lambda input, *dims: tuple(
        InputDim(i) for i in normalize_dims(tuple(dims), input.ndim)
    ),
    Tensor.transpose: lambda input, dim0, dim1: dim_transpose(input.ndim, dim0, dim1),
    Tensor.view: lambda input, *shape: view_groups(input.shape, shape),
}


def propagate_shape_and_sharding(
    input_src_placements: Sequence[Placement],
    local_in_shape: Shape,
    rule: DimMap,
    mesh_sizes: Shape,
) -> Tuple[Sequence[Placement], Sequence[Placement]]:
    """
    Determine input target sharding and output sharding based on
    given global tensor shape and input source sharding.

    Sharding propagation follows mapped dimensions:
    - An output dimension that maps directly to an input dimension is sharded equally
    - An output dimension that is a flattened set of input dimensions can only be
      sharded if only the leftmost flattened dimension is sharded.
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
    assert len(input_src_placements) == len(mesh_sizes)
    # for each input dim, for each mesh dim, provides a list of possible shardable dimensions
    mesh_ndim = len(mesh_sizes)
    shardable_dims: Dict[int, List[bool]] = {}

    # in case an input dimension disappears (e.g. collapsing, reduction)
    # we cannot shard in that dimension (we need a replication fall-back rule)
    seen_input_dims: Set[int] = set()

    def collect_used_inputs(cmd: DimSpec) -> None:
        if isinstance(cmd, InputDim):
            seen_input_dims.add(cmd.input_dim)
        for inp in cmd.inputs():
            collect_used_inputs(inp)

    for cmd in rule:
        collect_used_inputs(cmd)
    for dim in range(len(local_in_shape)):
        shardable_dims[dim] = [dim in seen_input_dims] * mesh_ndim

    def get_in_dim_to_shard(cmd: DimSpec) -> Optional[InputDim]:
        if isinstance(cmd, InputDim):
            return cmd
        elif isinstance(cmd, Flatten):
            for dim in cmd.input_dims[1:]:
                if isinstance(dim, InputDim):
                    shardable_dims[dim.input_dim] = [False] * mesh_ndim
            dim0 = cmd.input_dims[0]
            return dim0 if isinstance(dim0, InputDim) else None
        elif isinstance(cmd, Split):
            in_dim = get_in_dim_to_shard(cmd.input_dim)
            out_size = cmd.group_shape[cmd.split_id]
            if cmd.split_id == 0 and in_dim is not None:
                # we need to check that the input dimension is divisible
                # by the size of the submesh we're sharding it on
                # NOTE: it would be possible to shard the same input dimension
                # on more than one mesh dimension. In that case, the dimension
                # needs to be divisible by the product of mesh sizes.
                # In order to keep the problem more tractable, we will not consider
                # double resharding as a suggestion (e.g. [Shard(0), Shard(0) ])
                # but we will allow it if that's the input and it's compatible

                # 1. is this dimension shardable on each individual mesh dim?
                shardable_dims[in_dim.input_dim] = [
                    out_size % mesh_dim_size == 0 for mesh_dim_size in mesh_sizes
                ]

                # 2. here we special case things like [Shard(0), Shard(0)]
                submesh_size = 1
                for size, shard in zip(mesh_sizes, input_src_placements):
                    if isinstance(shard, Shard) and shard.dim == in_dim:
                        submesh_size *= size
                assert (
                    out_size % submesh_size == 0
                ), f"Resulting dimension size {out_size} is not divisible by its mesh dimension {submesh_size}."

            # we will only shard our first component of the split
            return in_dim if cmd.split_id == 0 else None
        elif isinstance(cmd, Repeat):
            in_dim = get_in_dim_to_shard(cmd.input_dim)
            if in_dim is not None:
                shardable_dims[in_dim.input_dim] = [False] * mesh_ndim
            return None
        else:
            return None

    # for each output dim, find the corresponding input dim in terms of sharding prop
    shard_dim_map = {}
    for dim, cmd in enumerate(rule):
        in_dim = get_in_dim_to_shard(cmd)
        if in_dim is not None:
            shard_dim_map[in_dim.input_dim] = dim

    input_tgt_placements = [
        (
            Replicate()
            if isinstance(p, Shard) and not shardable_dims[p.dim][mesh_dim]
            else p
        )
        for mesh_dim, p in enumerate(input_src_placements)
    ]
    output_placements = [
        Shard(shard_dim_map[p.dim]) if isinstance(p, Shard) else p
        for p in input_tgt_placements
    ]

    return input_tgt_placements, output_placements


def get_reshape_strategy(op):
    dim_map = dim_maps[op]

    # def reshape_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    def reshape_strategy(meta, mesh):
        assert meta.node.op == "call_method", "Node should have call_method op."
        args_schema = [meta["common"]["self"]] + [
            i for i in meta["common"]["args"].values()
        ]
        rules = dim_map(*args_schema)
        parent_node = meta.node.args[0]
        # input_strategy = cast(OpStrategy, op_schema.args_schema[0])
        input_strategy = parent_node.meta["mase"]["software"]["autosharding"][
            "op_strategy"
        ]
        global_in_shape = meta["common"]["self"].shape
        assert global_in_shape is not None, "Shape required."

        output_strategy = OpStrategy([])
        for input_placement_strategy in input_strategy.strategies:
            input_src_spec = input_placement_strategy.output_spec

            input_tgt_placements, output_placements = propagate_shape_and_sharding(
                input_src_spec.placements,
                tuple(global_in_shape),
                rules,
                mesh.mesh_shape,
            )

            # TODO: optimize this. we shouldn't simply blindly replicate
            #       unshardable dims ...
            # FIXME: this can be wrong for situations where we have
            #        [Shard(0), Shard(0)]
            input_tgt_spec = DTensorSpec(
                placements=tuple(input_tgt_placements),
                mesh=input_src_spec.mesh,
                tensor_meta=input_src_spec.tensor_meta,
            )

            output_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(output_placements),
                tensor_meta=TensorMeta(
                    shape=meta["common"]["results"]["data_out_0"]["shape"],
                    stride=None,
                    dtype=meta["common"]["results"]["data_out_0"]["torch_dtype"],
                ),
            )
            output_strategy.strategies.append(
                PlacementStrategy(
                    output_specs=output_spec,
                    input_specs=(input_tgt_spec,),
                )
            )

        return output_strategy

    return reshape_strategy
