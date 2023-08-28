import optuna

from functools import partial
from .base import StrategyBase

from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from chop.passes.analysis.total_bits_estimator import total_bits_module_analysis_pass


class StrategyOptunaAlgorithm(StrategyBase):
    iterative = False

    def read_setup(self):
        setup = self.config["setup"]
        self.n_jobs = setup["n_jobs"]
        self.n_trials = setup["n_trials"]
        self.timeout = setup["timeout"]
        self.sampler = setup["sampler"]
        self.model_parallel = setup["model_parallel"]
        self.runner_style = setup["runner_style"]
        self.runner = self.get_runner(self.runner_style)

    def sampler_map(self, name):
        match name.lower():
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "nsgaii":
                sampler = optuna.samplers.NSGAIISampler()
            case "nsgaiii":
                sampler = optuna.samplers.NSGAIIISampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case _:
                raise ValueError(f"Unknown sampler name: {name}")
        return sampler

    def compute_software_metric(self, model):
        # loss and a simple cost, this is a toy for testing
        loss, _ = self.runner(model)
        # TODO: add input associated cost?
        cost = total_bits_module_analysis_pass(model, {})
        return loss + cost["avg_bit"]

    def sample(self, indexes, quant_config_seed, search_space):
        sampled_config = search_space.config_sampler(
            indexes=indexes, config_seed=quant_config_seed
        )
        sampled_config = search_space.config_parser(
            sampled_config, search_space.get_num_layers()
        )
        model = search_space.rebuild_model(sampled_config)
        return self.compute_software_metric(model)

    def objective(self, trial, quant_config_seed, search_space):
        # wrapper for optuna Tiral
        sampled = {}
        for k, v in search_space.index_ranges.items():
            sampled[k] = trial.suggest_int(k, 0, v - 1)
        indexes = search_space.transform_flat_dict_to_nested_dict(sampled)
        return self.sample(indexes, quant_config_seed, search_space)

    def search(self, search_space):
        indexes, config_seed = search_space.build_opt_seed_and_indexes()
        study = optuna.create_study(sampler=self.sampler_map(self.sampler))
        study.optimize(
            func=partial(
                self.objective, quant_config_seed=config_seed, search_space=search_space
            ),
            n_trials=1,
        )
        best_config = study.best_params
        indexes = search_space.transform_flat_dict_to_nested_dict(best_config)
        sampled_config = search_space.config_sampler(indexes, config_seed=config_seed)
        sampled_config = search_space.config_parser(
            sampled_config, search_space.get_num_layers()
        )
        model = search_space.rebuild_model(sampled_config)
        return study.best_value, sampled_config, model
