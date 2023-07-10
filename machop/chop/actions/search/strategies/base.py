class StrategyBase:
    def __init__(self, config):
        self.config = config
        self.read_config()

    def read_config(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def feedback(self):
        raise NotImplementedError()
