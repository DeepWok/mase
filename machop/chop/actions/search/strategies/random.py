import random

from .base import StrategyBase


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


class StrategyRandom(StrategyBase):
    iterative = False

    def read_config(self):
        self.pool = self.config["pool"]

    def sampling(self, search_space):
        sampled = []
        for _ in range(self.pool):
            sampled.append(self.random_strategy(search_space))
        return sampled

    def random_strategy(self, search_space):
        sample = {}
        for n, v in search_space.items():
            index = random.randint(0, v - 1)
            sample[n] = index
        return sample

    def get_best(self, metrics, samples):
        min_pos = argmin(metrics)
        best_sample, best_metric = samples[min_pos], metrics[min_pos]
        return best_metric, best_sample

    def search(self, search_space, runner):
        sampled = self.sampling(search_space.search_space_flattened)
        samples = []
        for sample in sampled:
            samples.append(search_space.build_sample(sample)[0])

        metrics = []
        for sample in samples:
            metric = runner.get_performance(sample, search_space, runner.runner)
            metrics.append(metric)
        best_metric, best_sample = self.get_best(metrics, samples)
        best_mg = search_space.get_model(best_sample)
        return best_metric, best_sample, best_mg
