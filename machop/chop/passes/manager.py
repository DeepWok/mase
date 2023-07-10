class PassManager:
    def __init__(self, passes=None):
        if passes is None:
            self.passes = []
        else:
            self.passes = passes

    def add_pass(self, p):
        self.passes.append(p)

    def delete_pass(self, name=None, p=None):
        raise NotImplementedError

    def apply_passes(self):
        raise NotImplementedError
