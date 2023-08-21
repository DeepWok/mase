import torch

from .BaseInitializer import BaseInitializer, Memorize

from typing import Optional

from .utils.BinarizeSign import BinarizeSign
from .utils import truth_table


class BaseTrainer(torch.nn.Linear):
    """This class is the base class for trainers. provide a consistent interface for different ways of tables approximation."""

    binarization_level: int
    k: int
    kk: int
    input_expanded: bool
    tables_count: int
    initializer: BaseInitializer

    def __init__(
        self,
        tables_count: int,
        k: int,
        binarization_level: int,
        input_expanded: bool,
        device: Optional[str],
    ) -> None:
        """Initalize BaseTrainer common data structure.

        Args:
            tables_count (int): Number of tables consumers need to train
            k (int): Number of inputs for each table.
            binarization_level (int): which level of binarization is applied, 0 no binarization , 1 only weights binarized , 2 input also, and 3 output also binarized
            input_expanded (bool): If set to True, means all LUT's inputs are considered during calculations , else only the first input will considered and the remaining will be masked.
            device (str): device of the output tensor.
        """
        self.k = k
        self.kk = 2**k
        self.binarization_level = binarization_level
        self.input_expanded = input_expanded
        self.tables_count = tables_count

        super(BaseTrainer, self).__init__(
            in_features=self.kk,
            out_features=tables_count,
            bias=False,
            device=device,
        )
        self.set_input_expanded(input_expanded)

    def set_binarization_level(self, binarization_level: int) -> None:
        """binary calculations

        Args:
            binarization_level (int): which level of binarization is applied, 0 no binarization , 1 only weights binarized , 2 input also, and 3 output also binarized
        """
        self.binarization_level = binarization_level

    def set_input_expanded(self, input_expanded: bool) -> None:
        """Set the value for input expansion, either we use expanded input for not, using expanded input means we only consider first input for each lut.
           Please note that this not applicable if you are using minimal look up tables setup.

        Args:
            input_expanded (bool): boolean value of the new input_expanded.
        """
        self.input_expanded = input_expanded

        if not self.input_expanded:
            self.weight_mask = torch.zeros_like(self.weight)
            self.weight_mask[:, 0] = 1
            self.weight.weight_mask = self.weight_mask
        else:
            self.weight.weight_mask = None

    def update_grad_expanded(self) -> None:
        if not self.input_expanded:
            self.weight.grad = self.weight.grad * self.weight_mask

    def set_initializer(self, initializer: BaseInitializer) -> None:
        self.initializer = initializer

    def clear_initializion(self):
        if self.initializer is not None:
            self.initializer.clear()

    def update_initialized_weights(self):
        if self.initializer is not None:
            self.weight.data = self.initializer.update_luts_weights()

    def update_memoization(self):
        self.memorize


class LagrangeTrainer(BaseTrainer):
    device: Optional[str]
    truth_table: torch.Tensor

    def __init__(
        self,
        tables_count: int,
        k: int,
        binarization_level: int,
        input_expanded: bool,
        device: Optional[str],
    ):
        """Lagrange Approximation is using Lagrange interpolation to represent differentiable look-up tables.

        Args:
            tables_count (int): Number of tables consumers need to train
            k (int): Number of inputs for each table.
            binarization_level (int): which level of binarization is applied, 0 no binarization , 1 only weights binarized , 2 input also, and 3 output also binarized
            input_expanded (bool): If set to True, means all LUT's inputs are considered during calculations , else only the first input will considered and the remaining will be masked.
            device (str): device of the output tensor.
        """
        self.device = device
        self.truth_table = truth_table.generate_truth_table(k, 1, device)
        super(LagrangeTrainer, self).__init__(
            tables_count=tables_count,
            k=k,
            binarization_level=binarization_level,
            input_expanded=input_expanded,
            device=device,
        )
        self.gamma = torch.nn.Parameter(
            torch.tensor(1.0, requires_grad=True)
        )  # scaling factor

    def _validate_input(self, input: torch.tensor):
        """validate inputs dim before passing throw LUTs

        Args:
            input (torch.tensor): input from forward function.

        Raises:
            Exception: Invalid input dim
        """
        _rows_count = input.shape[-1]
        _tbl_count = int(_rows_count / self.k)
        if _rows_count % self.k != 0 or _tbl_count != self.tables_count:
            raise Exception("Invalid input dim")

    def forward(
        self,
        input: torch.tensor,  # [samples, table entries (table count * k)]
        targets: torch.tensor = None,
        initalize: bool = False,
    ) -> torch.Tensor:
        if initalize and self.initializer is not None and targets is not None:
            self.initializer.update_counter(input, targets)

        self._validate_input(input)
        input_truth_table = (
            input.view(-1, self.k, 1) * self.truth_table
        )  # [table_count * samples, k, 4]

        if self.binarization_level > 0:
            input_truth_table.data = input_truth_table.data.sign()

        if not self.input_expanded:
            input_truth_table *= -1
            reduced_table = input_truth_table[:, 0, :]
        else:
            input_truth_table = 1 + input_truth_table
            reduced_table = input_truth_table.prod(
                dim=-2
            )  # [tables_count*samples, self.kk]
        reduced_table = reduced_table.view(
            -1, self.tables_count, self.kk
        )  # [samples, tables_count, self.kk]
        if self.binarization_level > 0:
            _weight = BinarizeSign.apply(self.weight)  # [tables_count, self.kk]
        else:
            _weight = self.weight
        if not self.input_expanded:
            _weight = _weight * self.weight_mask

        out = reduced_table * _weight * self.gamma.abs()
        return out.sum(-1)

    # if we have more intitalizers , may be better we introduce builders for each base module , where we all the object creation logic should live.
    def set_memorize_as_initializer(self) -> None:
        initializer = Memorize(self.tables_count, self.k, self.kk, self.device)
        self.set_initializer(initializer=initializer)
