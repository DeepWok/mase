import numpy as np


class MaskBase:
    k: int
    table_input_selections: list
    replace: bool

    def __init__(
        self, k: int, table_input_selections: list, replace: bool = False
    ) -> None:
        """MaskBase represents the base class for mask builders. Mask builders are consumed by high-level layers (Linear, Conv2d .. etc)
           to build a mask that wire layer input the LUT inputs in the trainer.

        Args:
            k (int): k degree of each table
            table_input_selections (list): List of tuple(selector,list(selector)) , where selector is 1d Array that select an element in the input. The second element in the tuple
            represents the elements we select from to fill in the k-1 inputs for the table.
            replace (bool, optional): Either we select from the list with/out a replacement.
        """
        self.k = k
        self.table_input_selections = table_input_selections
        self.replace = replace
        super().__init__()

    def build(self):
        return

    def get_tables_count(self):
        return


class MaskExpanded(MaskBase):
    def __init__(
        self, k: int, table_input_selections: list, replace: bool = False
    ) -> None:
        """MaskExpanded : Implements mask expansion where each input is expanded in a LUT and we randomly select the remaining inputs for each LUT from table_input_selections list.

        Args:
            k (int): k degree of each table
            table_input_selections (list): List of tuple(selector,list(selector)) , where selector is 1d Array that select an element in the input. The second element in the tuple
            represents the elements we select from to fill in the k-1 inputs for the table.
            replace (bool, optional): Either we select from the list with/out the replacement.
        """
        super().__init__(
            k=k, table_input_selections=table_input_selections, replace=replace
        )

    def build(self) -> np.ndarray:
        """build expanded mask where for each input we have a table and remaining inputs for the same table we select in random from the given selection list.

        Returns:
            np.ndarray: expansion mask.
        """
        result = []
        for table_input_selection in self.table_input_selections:
            idx = table_input_selection[0]
            possible_selections = table_input_selection[1]
            result.append(idx)
            ids = np.random.choice(
                len(possible_selections), self.k - 1, replace=self.replace
            )
            for id in ids:
                result.append(possible_selections[id])
        return np.array(result)

    def get_tables_count(self) -> int:
        input_length = len(self.table_input_selections)
        return input_length
