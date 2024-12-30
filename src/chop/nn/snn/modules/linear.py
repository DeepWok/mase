from torch import nn
import chop.nn.snn.base as base


class Linear(nn.Linear, base.StepModule):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, step_mode="s"
    ) -> None:
        """
        * :ref:`API in English <Linear-en>`
        .. _Linear-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Linear` for other parameters' API
        """
        super().__init__(in_features, out_features, bias)
        self.step_mode = step_mode
