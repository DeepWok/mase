import torch

STAT_NAME_TO_CLS = {}


def _add_to_stat_mapping(cls):
    global STAT_NAME_TO_CLS
    assert issubclass(cls, _StatBase)
    STAT_NAME_TO_CLS[cls.stat_name] = cls


class _StatBase:
    stat_name: str

    def __init__(self) -> None:
        pass

    def update_a_sample(self, *args, **kwargs):
        raise NotImplementedError

    def finalize(self) -> dict:
        raise NotImplementedError

    def finalize_to_list(self) -> dict:
        return NotImplementedError

    def export(self):
        return {self.stat_name: self.finalize()}

    def export_to_list(self):
        return {self.stat_name: self.finalize_to_list()}


@_add_to_stat_mapping
class Record(_StatBase):
    stat_name = "record"

    def __init__(self, add_new_dim_before_concat) -> None:
        super().__init__()
        self.add_new_dim = add_new_dim_before_concat
        self.data = None
        self.count = None
        self.total_size_in_bytes = None

    def update_a_sample(self, new_s: torch.Tensor):
        new_s = new_s.clone()
        assert isinstance(new_s, torch.Tensor)
        if self.add_new_dim:
            new_s = new_s.unsqueeze(0)
        if self.data is None:
            self.data = new_s
            self.count = new_s.shape[0]
        else:
            self.data = torch.concat((self.data, new_s), dim=0)
            self.count += new_s.shape[0]
        self.total_size_in_bytes = self.data.element_size() * self.data.nelement()

    def finalize(self) -> dict:
        return {"data": self.data}

    def finalize_to_list(self) -> dict:
        return {"data": self.data.tolist()}


@_add_to_stat_mapping
class Variance(_StatBase):
    """
    Use Welford's online algorithm to calculate running variance and mean
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    stat_name = "variance"

    def __init__(self, existing_count=0, existing_mean=0.0, existing_m2=0.0) -> None:
        super().__init__()
        assert isinstance(existing_count, int)
        assert isinstance(existing_mean, (float, torch.Tensor))
        assert isinstance(existing_m2, (float, torch.Tensor))
        self.count: int = existing_count
        self.mean: torch.Tensor = existing_mean
        self.m2: torch.Tensor = existing_m2

    def update_a_sample(self, new_s: torch.Tensor):
        new_s = new_s.clone()
        self.count += 1
        delta = new_s - self.mean
        self.mean += delta / self.count
        delta2 = new_s - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> dict:
        if self.count < 2:
            result = {"mean": "NA", "variance": "NA", "sample_variance": "NA"}
        else:
            result = {
                "count": self.count,
                "mean": self.mean,
                "variance": self.m2 / self.count,
                "sample_variance": self.m2 / (self.count - 1),
            }
        return result

    def finalize_to_list(self) -> dict:
        if self.count < 2:
            result = {"mean": "NA", "variance": "NA", "sample_variance": "NA"}
        else:
            result = {
                "count": self.count,
                "mean": self.mean.tolist(),
                "variance": (self.m2 / self.count).tolist(),
                "sample_variance": (self.m2 / (self.count - 1)).tolist(),
            }
        return result

    def end_aggregate(self):
        assert self.count >= 2
        return {
            "existing_count": self.count,
            "existing_mean": self.mean,
            "existing_m2": self.m2,
        }
