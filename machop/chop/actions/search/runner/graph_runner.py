from .base import SearchRunnerBase


class SearchRunnerMG(SearchRunnerBase):
    def get_performance(self, sample, search_space, runner):
        mg = search_space.get_model(sample)
        metric = runner(mg.model)
        return metric
