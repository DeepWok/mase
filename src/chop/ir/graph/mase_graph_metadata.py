class MaseGraphMetadata:
    def __init__(self, graph):
        self.graph = graph
        self.parameters = {
            "common": {},
            "software": {},
            "hardware": {},
        }

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value
