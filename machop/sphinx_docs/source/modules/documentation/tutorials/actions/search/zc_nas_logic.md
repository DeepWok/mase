# How does zero cost search work ?

#### Class `ZeroCostProxy(SearchSpaceBase)`

Build new search space class located at `chop/actions/search/search_space/zero_cost/graph.py`. Revise the `build_search_space()` and `rebuild_model()` functions.

- `build_search_space()`: Return the dictionary format of the search options
- `rebuild_model()`: Given the option combination, return the corresponding neural architecture and post-trained metrics from NAS-Bench-201.
    ```python
  def rebuild_model(self, sampled_config, is_eval_mode: bool = False):
        """
        Group 2 NAS-Proxy
        This method rebuilds the model based on the sampled configuration. It also sets the model to evaluation or training mode based on the is_eval_mode parameter.
        It queries the NAS-Bench-201 API for the architecture performance and uses the returned architecture to rebuild the model.
        """

        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        if "nas_zero_cost" in sampled_config:
            nas_config = generate_configs(sampled_config["nas_zero_cost"])
            nasbench_dataset = sampled_config["nas_zero_cost"]["dataset"]
        else:
            nas_config = generate_configs(sampled_config["default"])
            nasbench_dataset = sampled_config["default"]["dataset"]

        arch = nas_config["arch_str"]
        index = api.query_index_by_arch(arch)
        results = api.query_by_index(index, nasbench_dataset)
        data = api.get_more_info(index, nasbench_dataset)

        model_arch = get_cell_based_tiny_net(nas_config)
        model_arch = model_arch.to(self.accelerator)

        return model_arch, data
    ```

#### Class `RunnerZeroCost(SWRunnerBase)`
Build new software runner for zero cost proxy values calculation located at `/chop/actions/search/strategies/runners/software/zc.py`. Given the neural architecture and needed zero-cost proxies, return the zero-cost proxy values.

Revise the `__call__` function as follows.
```python
def __call__(self, data_module, model, sampled_config) -> dict[str, float]:

        zero_cost_metrics = {}

        data_loader = self.config["data_loader"]
        metric_names = self.config["metrics"]

        if data_loader == "train_dataloader":
            dataloader = data_module.train_dataloader()
        elif data_loader == "val_dataloader":
            dataloader = data_module.val_dataloader()

        dataload_info = ("random", 1, 10)
        device = self.accelerator

        for metric_name in metric_names:
            if metric_name in self.available_metrics:
                zero_cost_metrics[metric_name] = find_measures(
                    model,
                    dataloader,
                    dataload_info,  # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                    device,
                    loss_fn=F.cross_entropy,
                    measure_names=[metric_name],
                    measures_arr=None,
                )[metric_name]
            else:
                raise ValueError(
                    "Zero cost metrics should be chosen from ['fisher', 'grad_norm', 'grasp', 'l2_norm', 'plain', 'snip', 'synflow', 'naswot', 'naswot_relu', 'tenas', 'zico']!!!"
                )

        return zero_cost_metrics
```

Then revise the `objective` method in `chop/actions/search/strategies/optuna.py` to adjust the newly-designed search space and runner.