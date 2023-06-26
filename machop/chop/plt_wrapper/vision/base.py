from ..base import WrapperBase


class VisionModelWrapper(WrapperBase):
    def __init__(
        self, model, info=None, learning_rate=5e-4, epochs=200, optimizer=None
    ):
        super().__init__(
            model=model,
            info=info,
            learning_rate=learning_rate,
            epochs=epochs,
            optimizer=optimizer,
        )
