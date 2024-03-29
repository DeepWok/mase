from ..base import WrapperBase


class VisionModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info=None,
        learning_rate=5e-4,
        weight_decay=0,
        epochs=200,
        optimizer=None,
        batch_size=128, # self added
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            optimizer=optimizer,
            batch_size=batch_size,
        )
