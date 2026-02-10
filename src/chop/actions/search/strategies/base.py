import joblib

from .runners import get_sw_runner


class SearchStrategyBase:
    """
    Base class for search strategies.

    ---

    What is a search strategy?

    search_strategy is responsible for:
    - setting up the data loader
    - perform search, which includes:
        - sample the search_space.choice_lengths_flattened to get the indexes
        - call search_space.rebuild_model to build a new model with the sampled indexes
        - calculate the software metrics through the configured runners
    - save the results

    ---
    To implement a new search strategy, you need to implement the following methods:
    - `_post_init_setup(self) -> None`: additional setup for the subclass instance
    - `search(self, search_space) -> Any: perform search, and save the results

    ---
    Check `machop/chop/actions/search/strategies/optuna.py` for an example.
    """

    is_iterative: bool = None  # whether the search strategy is iterative or not

    def __init__(
        self,
        model_info,
        data_module,
        dataset_info,
        task: str,
        config: dict,
        accelerator,
        save_dir,
        visualizer,
    ):
        self.dataset_info = dataset_info
        self.task = task
        self.config = config
        self.accelerator = accelerator
        self.save_dir = save_dir
        self.data_module = data_module
        self.visualizer = visualizer

        self.sw_runner = []
        # the software runner's __call__ will use the rebuilt model to calculate the software metrics like accuracy, loss, ...

        if "sw_runner" in config.keys():
            for runner_name, runner_cfg in config["sw_runner"].items():
                self.sw_runner.append(
                    get_sw_runner(
                        runner_name,
                        model_info,
                        task,
                        dataset_info,
                        accelerator,
                        runner_cfg,
                    )
                )

        # Keep an empty attribute for backwards compatibility with downstream
        # strategy code paths that still reference `self.hw_runner`.
        self.hw_runner = []
        if "hw_runner" in config.keys():
            raise ValueError(
                "`hw_runner` is no longer supported. Remove hardware runner config."
            )

        self._post_init_setup()

    @staticmethod
    def _save_study(study, save_path):
        """
        Save the study object. The subclass can call this method to save the study object at the end of the search.
        """
        with open(save_path, "wb") as f:
            joblib.dump(study, f)

    def _post_init_setup(self):
        """
        Post init setup. This is where additional config parsing and setup should be done for the subclass instance.
        """
        raise NotImplementedError()

    def search(self, search_space):
        """
        Perform search, and save the results.
        """
        raise NotImplementedError()
