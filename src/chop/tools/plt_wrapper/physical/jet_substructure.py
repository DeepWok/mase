from ..base import WrapperBase


class JetSubstructureModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info=None,
        learning_rate=5e-4,
        weight_decay=0,
        scheduler_args=None,
        epochs=200,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_args=None,
            epochs=epochs,
            optimizer=optimizer,
        )
