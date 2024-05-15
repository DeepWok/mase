import math
import logging
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from numpy import ndarray

logger = logging.getLogger(__name__)

STAT_NAME_TO_CLS = {}


def _add_to_stat_mapping(cls):
    global STAT_NAME_TO_CLS
    assert issubclass(cls, _StatBase)
    STAT_NAME_TO_CLS[cls.name] = cls
    return cls


class _StatBase:
    name: str = None

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def update_a_sample(self, *args, **kwargs) -> None:
        """
        Update the stat with a new sample.

        If the sample is a Tensor, it will be detached and copied to the device specified in the constructor.
        If the sample is a list, tuple, int, or float, it will be converted to a Tensor.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self) -> dict[str, Tensor]:
        """
        Compute/finalize the stat and return a dict of results.

        The results should be a dict of Tensors.
        """
        raise NotImplementedError

    def export(self) -> dict[str, dict[str, list]]:
        """
        Export the stat to a dict of dict of lists.

        This method calls compute() and converts the results to lists, which is more friendly to toml serialization.
        """
        results = self.compute()
        return {
            self.name: {
                k: v.tolist() if isinstance(v, (Tensor)) else v
                for k, v in results.items()
            }
        }

    def __repr__(self) -> str:
        return type(self).__name__.capitalize()


@_add_to_stat_mapping
class Record(_StatBase):
    """
    Record all samples passed in

    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        add_new_dim_before_concat (bool): if True, add a new dimension before concatenating the samples.
    """

    name = "record"

    def __init__(self, device=None, add_new_dim_before_concat: bool = False) -> None:
        super().__init__()
        self.device = device
        self.add_new_dim: bool = add_new_dim_before_concat
        self.data: Tensor = None
        self.count: int = None
        self.total_size_in_bytes: float = None

    @torch.no_grad()
    def update_a_sample(self, new_s: Tensor):
        if isinstance(new_s, (list, tuple, int, float)):
            new_s = torch.tensor(new_s).float()
        assert isinstance(new_s, Tensor)
        new_s = new_s.clone().detach().float()
        if self.device is not None:
            new_s = new_s.to(self.device)
        if self.add_new_dim:
            new_s = new_s.unsqueeze(0)
        if self.data is None:
            self.data = new_s
            self.count = 1
        else:
            self.data = torch.concat((self.data, new_s), dim=0)
            self.count += 1
        self.total_size_in_bytes = self.data.element_size() * self.data.nelement()

    @torch.no_grad()
    def compute(self) -> dict:
        return {
            "data": self.data,
            "count": self.count,
            "size_in_bytes": self.total_size_in_bytes,
        }


@_add_to_stat_mapping
class VarianceOnline(_StatBase):
    """
    Use Welford's online algorithm to calculate running variance and mean

    This saves memory by not storing all the samples, but the variance is not precise when the count is small.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    ---
    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        dims (str|list|None): the dimensions to reduce. If "all", reduce all dimensions. If None, do not reduce any dimension. If a list, reduce the specified dimensions.
    """

    name = "variance_online"

    def __init__(self, device=None, dims: Literal["all"] | None | list = "all") -> None:
        super().__init__()
        self.device = device
        if isinstance(dims, (list, tuple)):
            # assert sorted(dims) == list(
            #     range(min(dims), max(dims) + 1)
            # ), "dims must be consecutive"
            self.dims_to_reduce = sorted(dims)
        else:
            assert dims in ["all", None]
            self.dims_to_reduce = dims

        self.count: int = 0
        self.mean: Tensor = 0
        self.m: Tensor = 0

    @staticmethod
    def _update(
        new_s: Tensor,
        count: int,
        mean: Tensor,
        m: Tensor,
    ):
        count += 1
        delta = new_s - mean
        mean += delta / count
        m += delta * (new_s - mean)

        return count, mean, m

    @staticmethod
    def _reshape_a_sample(new_s: Tensor, dims_to_reduce: list[int]):
        dims_to_keep = [i for i in range(new_s.ndim) if i not in dims_to_reduce]
        transpose_dims = dims_to_keep + dims_to_reduce
        new_s = new_s.permute(*transpose_dims)
        new_s = torch.flatten(new_s, start_dim=len(dims_to_keep), end_dim=-1)
        return new_s

    @torch.no_grad()
    def update_a_sample(self, new_s: Tensor):
        if isinstance(new_s, (list, tuple, int, float)):
            new_s = torch.tensor(new_s)
        assert isinstance(new_s, Tensor)
        new_s = new_s.clone().detach().float()

        if self.device is not None:
            new_s = new_s.to(self.device)

        match self.dims_to_reduce:
            case "all":
                new_s = torch.flatten(new_s)
                n_b = new_s.nelement()
                mean_b = new_s.mean()

                delta = mean_b - self.mean
                self.mean += delta * n_b / (self.count + n_b)
                self.m += new_s.var() * n_b + delta**2 * self.count * n_b / (
                    self.count + n_b
                )
                self.count += n_b
            case None:
                self.count, self.mean, self.m = self._update(
                    new_s=new_s,
                    count=self.count,
                    mean=self.mean,
                    m=self.m,
                )
            case _:
                # self.dims_to_reduce is a list
                new_s = self._reshape_a_sample(
                    new_s, dims_to_reduce=self.dims_to_reduce
                )
                for i in range(new_s.size(-1)):
                    self.count, self.mean, self.m = self._update(
                        new_s=new_s[..., i],
                        count=self.count,
                        mean=self.mean,
                        m=self.m,
                    )

    @torch.no_grad()
    def compute(self) -> dict:
        if self.count < 2:
            logger.warning(
                f"VarianceOnline: count is {self.count}, which is less than 2. "
                "Returning NA for mean and variance."
            )
            return {"mean": "NA", "variance": "NA"}

        var = self.m / self.count
        return {
            "mean": self.mean,
            "variance": var,
            "count": self.count,
        }


@_add_to_stat_mapping
class VariancePrecise(Record):
    """
    Concatenate samples and use torch.var, torch.mean to calculate variance and mean

    This is precise but may use massive memory when the count or sample size is large.

    ---
    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        dims (str|list|None): the dimensions to reduce. If "all", reduce all dimensions. If None, do not reduce any dimension. If a list, reduce the specified dimensions.
    """

    name = "variance_precise"

    def __init__(self, device=None, dims: Literal["all"] | None | list = "all") -> None:
        super().__init__(device=device, add_new_dim_before_concat=True)
        self.dims_to_reduce = dims

    @torch.no_grad()
    def update_a_sample(self, new_s: Tensor):
        super().update_a_sample(new_s=new_s)

    @torch.no_grad()
    def compute(self) -> dict[str, ndarray]:
        match self.dims_to_reduce:
            case "all":
                var = torch.var(self.data)
                mean = torch.mean(self.data)
                count = self.data.nelement()
            case None:
                count = self.data.size(0)
                if self.data.size(0) < 2:
                    logger.warning(
                        f"VariancePrecise: count is {self.data.size(0)}, which is less than 2. "
                        "Returning NA for mean and variance."
                    )
                    mean = "NA"
                    var = "NA"
                else:
                    var = torch.var(self.data, dim=0)
                    mean = torch.mean(self.data, dim=0)
            case _:
                dims_to_reduce = [i + 1 for i in self.dims_to_reduce]
                count = self.data[dims_to_reduce].nelements()
                if self.data[dims_to_reduce].nelements() < 2:
                    logger.warning(
                        f"VariancePrecise: count is {self.data[dims_to_reduce].nelements()}, which is less than 2. "
                        "Returning NA for mean and variance."
                    )
                    var = "NA"
                    mean = "NA"
                else:
                    var = torch.var(self.data, dim=[0] + dims_to_reduce)
                    mean = torch.mean(self.data, dim=[0] + dims_to_reduce)
        return {"mean": mean, "variance": var, "count": count}


@_add_to_stat_mapping
class RangeNSigma(VarianceOnline, VariancePrecise):
    """
    Calculate the range of samples within n sigma based on the mean and variance.

    This stat assumes the samples are normally distributed.

    ---
    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        dims (str|list|None): the dimensions to reduce. If "all", reduce all dimensions. If None, do not reduce any dimension. If a list, reduce the specified dimensions.
        abs (bool): if True, take the absolute value of the samples before calculating the mean and variance.
        var_mode (str): "precise" or "online". If "precise", use VariancePrecise to calculate the variance. If "online", use VarianceOnline to calculate the variance.
        num_sigma (int): the number of sigma to use to calculate the range.
    """

    name = "range_n_sigma"

    def __init__(
        self,
        device=None,
        dims: Literal["all"] | None | list = "all",
        abs: bool = False,
        var_mode: Literal["precise"] | Literal["online"] = "online",
        num_sigma: int = 3,
    ) -> None:
        VarianceOnline.__init__(self, device=device, dims=dims)
        VariancePrecise.__init__(self, device=device, dims=dims)
        assert var_mode in ["precise", "online"]
        self.var_mode = var_mode
        self.abs = abs
        self.num_sigma: int = num_sigma
        self.min: Tensor = None
        self.max: Tensor = None

    @torch.no_grad()
    def update_a_sample(self, new_s: Tensor):
        if isinstance(new_s, (list, tuple, int, float)):
            new_s = torch.tensor(new_s).float()
        if self.abs:
            new_s = torch.abs(new_s)
        if self.var_mode == "online":
            VarianceOnline.update_a_sample(self, new_s=new_s)
        else:
            VariancePrecise.update_a_sample(self, new_s=new_s)

    @torch.no_grad()
    def compute(self) -> dict:
        if self.var_mode == "online":
            var_results = VarianceOnline.compute(self)
        else:
            var_results = VariancePrecise.compute(self)

        count = var_results["count"]
        if var_results["variance"] == "NA" or var_results["mean"] == "NA":
            logger.warning(
                f"RangeNSigma: count is {self.count}, which is less than 2. "
                "Returning NA for min and max."
            )
            return {"min": "NA", "max": "NA", "range": "NA", "count": count}
        else:
            sigma = math.sqrt(var_results["variance"])
            self.min = var_results["mean"] - self.num_sigma * sigma
            self.max = var_results["mean"] + self.num_sigma * sigma
            d_range = self.max - self.min
            return {"min": self.min, "max": self.max, "range": d_range, "count": count}


@_add_to_stat_mapping
class RangeMinMax(_StatBase):
    """
    Calculate the range of samples based on the min and max values.

    ---
    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        dims (str|list|None): the dimensions to reduce. If "all", reduce all dimensions. If None, do not reduce any dimension. If a list, reduce the specified dimensions.
        abs (bool): if True, take the absolute value of the samples before calculating the min and max.
    """

    name = "range_min_max"

    def __init__(
        self, device=None, dims: Literal["all"] | list | None = "all", abs: bool = False
    ) -> None:
        super().__init__()
        self.device = device
        self.dims = dims
        self.abs = abs
        self.min = None
        self.max = None
        self.count = 0

    @torch.no_grad()
    def update_a_sample(self, new_s: Tensor):
        if isinstance(new_s, (list, tuple, int, float)):
            new_s = torch.tensor(new_s).float()
        new_s = new_s.clone().detach().float()
        if self.device:
            new_s = new_s.to(self.device)

        if self.abs:
            new_s = torch.abs(new_s)

        if self.min is None:
            match self.dims:
                case None:
                    self.min = new_s
                    self.max = new_s
                    self.count += 1
                case "all":
                    self.min = torch.min(new_s)
                    self.max = torch.max(new_s)
                    self.count += new_s.nelement()
                case _:
                    n_elem = 1
                    for dim in self.dims:
                        n_elem *= new_s.size(dim)
                        self.min = torch.min(new_s, dim=dim)
                        self.max = torch.max(new_s, dim=dim)
                    self.count += n_elem
        else:
            match self.dims:
                case None:
                    self.min = torch.min(self.min, new_s)
                    self.max = torch.max(self.max, new_s)
                    self.count += 1
                case "all":
                    self.min = torch.min(self.min, torch.min(new_s))
                    self.max = torch.max(self.max, torch.max(new_s))
                    self.count += new_s.nelement()
                case _:
                    n_elem = 1
                    for dim in self.dims:
                        n_elem *= new_s.size(dim)
                        self.min = torch.min(self.min, torch.min(new_s, dim=dim))
                        self.max = torch.max(self.max, torch.max(new_s, dim=dim))
                    self.count += n_elem

    def compute(self) -> dict:
        if self.count < 2:
            logger.warning(
                f"RangeMinMax: count is {self.count}, which is less than 2. "
                "Returning NA for min and max."
            )
            minimum = "NA"
            maximum = "NA"
            d_range = "NA"
        else:
            minimum = self.min
            maximum = self.max
            d_range = self.max - self.min
        return {"min": self.min, "max": self.max, "range": d_range, "count": self.count}


@_add_to_stat_mapping
class RangeQuantile(Record):
    """
    Calculate the range of samples based on the quantile.

    ---
    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        dims (str|list|None): the dimensions to reduce. If "all", reduce all dimensions. If None, do not reduce any dimension. If a list, reduce the specified dimensions.
        abs (bool): if True, take the absolute value of the samples before calculating the min and max.
        quantile (float): the quantile to use to calculate the range.
    """

    name = "range_quantile"

    def __init__(
        self,
        device=None,
        abs: bool = False,
        dims: Literal["all"] | None | list = "all",
        quantile: float = 0.975,
    ) -> None:
        super().__init__(device=device, add_new_dim_before_concat=True)
        self.quantile = quantile
        self.abs = abs
        self.dims = dims

    @staticmethod
    def _permute_and_flatten(data: Tensor, dims_to_reduce: list[int]):
        dims_to_keep = [i for i in range(data.ndim) if i not in dims_to_reduce]
        transpose_dims = dims_to_keep + dims_to_reduce
        data = data.permute(*transpose_dims)
        data = torch.flatten(data, start_dim=len(dims_to_keep), end_dim=-1)
        return data

    @torch.no_grad()
    def compute(self) -> dict:
        if self.abs:
            self.data = torch.abs(self.data)

        match self.dims:
            case None:
                count = self.data.size(0)
                if count < 2:
                    logger.warning(
                        f"RangeQuantile: count is {self.data.size(0)}, which is less than 2. "
                        "Returning NA for min and max."
                    )
                    minimum = "NA"
                    maximum = "NA"
                    d_range = "NA"
                else:
                    minimum = self.data.quantile(1 - self.quantile, dim=0)
                    maximum = self.data.quantile(self.quantile, dim=0)
                    d_range = maximum - minimum
            case "all":
                minimum = self.data.quantile(1 - self.quantile)
                maximum = self.data.quantile(self.quantile)
                d_range = maximum - minimum
                count = self.data.nelement()
            case _:
                dims_to_reduce = [i + 1 for i in self.dims]

                self.data = self._permute_and_flatten(
                    data=self.data, dims_to_reduce=[0] + dims_to_reduce
                )
                minimum = self.data.quantile(1 - self.quantile, dim=-1)
                maximum = self.data.quantile(self.quantile, dim=-1)
                count = self.data.size(-1)
                d_range = maximum - minimum

        return {"min": minimum, "max": maximum, "range": d_range, "count": count}


@_add_to_stat_mapping
class AbsMean(_StatBase):
    """
    Online algorithm to compute the mean, taking absolute value first i.e. E(|x|)

    ---
    Args:
        device (str|None): the device to move the samples to. If None, the samples will not be moved.
        dims (str|list|None): the dimensions to reduce. If "all", reduce all dimensions. If None, do not reduce any dimension. If a list, reduce the specified dimensions.
        abs (bool): if True, take the absolute value of the samples before calculating the min and max.
    """

    # TODO Dimension reduction, device?

    name = "abs_mean"

    def __init__(
        self,
        device=None,
        dims: Literal["all"] | None | list = "all",
    ) -> None:
        super().__init__()
        self.device = device
        if isinstance(dims, (list, tuple)):
            # assert sorted(dims) == list(
            #     range(min(dims), max(dims) + 1)
            # ), "dims must be consecutive"
            self.dims_to_reduce = sorted(dims)
        else:
            assert dims in ["all", None]
            self.dims_to_reduce = dims

        self.count: int = 0
        self.mean: Tensor = 0

    @staticmethod
    def _update(
        new_s: Tensor,
        count: int,
        mean: Tensor,
    ):
        count += 1
        mean = (1 / count) * ((count - 1) * mean + abs(new_s))

        return count, mean

    @staticmethod
    def _reshape_a_sample(new_s: Tensor, dims_to_reduce: list[int]):
        dims_to_keep = [i for i in range(new_s.ndim) if i not in dims_to_reduce]
        transpose_dims = dims_to_keep + dims_to_reduce
        new_s = new_s.permute(*transpose_dims)
        new_s = torch.flatten(new_s, start_dim=len(dims_to_keep), end_dim=-1)
        return new_s

    @torch.no_grad()
    def update_a_sample(self, new_s: Tensor):
        if isinstance(new_s, (list, tuple, int, float)):
            new_s = torch.tensor(new_s)
        assert isinstance(new_s, Tensor)
        new_s = new_s.clone().detach().float()

        if self.device is not None:
            new_s = new_s.to(self.device)

        match self.dims_to_reduce:
            case "all":
                new_s = torch.flatten(new_s)
                n_b = new_s.nelement()
                mean_b = new_s.abs().mean()

                self.mean = (
                    1 / (self.count + n_b) * (self.count * self.mean + n_b * mean_b)
                )
                self.count += n_b
            case None:
                self.count, self.mean = self._update(
                    new_s=new_s,
                    count=self.count,
                    mean=self.mean,
                )
            case _:  # TODO other dims
                new_s = self._reshape_a_sample(
                    new_s, dims_to_reduce=self.dims_to_reduce
                )
                for i in range(new_s.size(-1)):
                    self.count, self.mean = self._update(
                        new_s=new_s[..., i],
                        count=self.count,
                        mean=self.mean,
                    )

    @torch.no_grad()
    def compute(self) -> dict:
        if self.count < 2:
            logger.warning(
                f"AbsMean: count is {self.count}, which is less than 2. "
                "Returning NA for mean."
            )
            return {"abs_mean": "NA"}

        return {
            "abs_mean": self.mean,
            "count": self.count,
        }


def create_new_stat(stat_name: str, **stat_kwargs):
    global STAT_NAME_TO_CLS
    assert (
        stat_name in STAT_NAME_TO_CLS
    ), f"Unknown stat name: {stat_name}. Available stat names: {list(STAT_NAME_TO_CLS.keys())}"
    stat_cls = STAT_NAME_TO_CLS[stat_name]
    return stat_cls(**stat_kwargs)
