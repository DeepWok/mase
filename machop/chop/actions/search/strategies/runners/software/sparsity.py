from chop.actions.search.strategies.runners.software.base import SWRunnerBase
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis.add_metadata.add_common_metadata import add_common_metadata_analysis_pass
from chop.passes.graph.analysis.add_metadata.add_software_metadata import add_software_metadata_analysis_pass
from chop.passes.graph.analysis.pruning.calculate_sparsity import add_pruning_metadata_analysis_pass
from chop.passes.graph.utils import get_mase_op


class RunnerPrunedSparsity(SWRunnerBase):
    available_metrics = ("weight_sparsity", "activation_sparsity")

    def __init__(self, model_info, task: str, dataset_info, accelerator, config: dict = None):
        super().__init__(model_info, task, dataset_info, accelerator, config)

    def _post_init_setup(self) -> None:
        pass

    def __call__(self, data_module, model: MaseGraph, sampled_config) -> dict[str, float]:
        assert isinstance(model, MaseGraph)
        element_data = {}

        # dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}
        # model, _ = add_common_metadata_analysis_pass(model, {"dummy_in": dummy_in, "force_device_meta": False})
        # model, _ = add_software_metadata_analysis_pass(model, None)
        # model, _ = add_pruning_metadata_analysis_pass(model, {"dummy_in": dummy_in, "add_value": True})


        total_weights = 0
        total_activations = 0
        pruned_weights = 0
        pruned_activations = 0
        sparsity_data = {}

        for node in model.fx_graph.nodes:
            if get_mase_op(node) in ["linear", "conv2d", "conv1d"]:
                meta = node.meta["mase"]
                num_weights = meta.parameters["common"]["args"]["parametrizations.weight.original"]["value"].numel()
                num_activations = meta.parameters["common"]["args"]["data_in_0"]['value'].numel()
                element_data[node.target] = {
                    "weights": num_weights,
                    "activations": num_activations,
                }
                sparsity_data[node.target] = {
                    "weight_sparsity": meta.parameters["software"]["args"]["weight"]["sparsity"],
                    "activation_sparsity": meta.parameters["software"]["args"]["data_in_0"]["sparsity"]
                }

        for node in element_data:
            total_weights += element_data[node]["weights"]
            total_activations += element_data[node]["activations"]

            pruned_weights += element_data[node]["weights"] * sparsity_data[node]["weight_sparsity"]
            pruned_activations += element_data[node]["activations"] * sparsity_data[node]["activation_sparsity"]

        overall_weight_sparsity = pruned_weights / total_weights
        overall_activation_sparsity = pruned_activations / total_activations

        return {
            "weight_sparsity": overall_weight_sparsity,
            "activation_sparsity": overall_activation_sparsity
        } 