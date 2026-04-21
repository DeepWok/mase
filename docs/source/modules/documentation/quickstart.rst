Quickstart
=============================

This page gives a brief overview of the main Mase workflows.
For hands-on walkthroughs, follow the :doc:`tutorials`.

Importing a model
-----------------

To import a model into Mase, wrap it in a ``MaseGraph``. Mase uses Torch FX to build a
computation graph that analysis and transform passes can iterate over.

.. code-block:: python

    from transformers import AutoModelForSequenceClassification
    from chop import MaseGraph
    import chop.passes as passes

    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    mg = MaseGraph(model, hf_input_names=["input_ids", "attention_mask", "labels"])
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(mg)

See :doc:`tutorials/tutorial_1_introduction_to_mase_script` for a full walkthrough.

Architecture Search
-------------------

Mase integrates with `Optuna <https://optuna.org>`_ to search over model architecture dimensions
(number of layers, heads, hidden size) and layer type choices (e.g. ``Linear`` vs ``Identity``).

.. code-block:: python

    import optuna
    from optuna.samplers import RandomSampler
    from chop.tools import get_tokenized_dataset, get_trainer

    dataset, tokenizer = get_tokenized_dataset("imdb", "bert-base-uncased", return_tokenizer=True)

    def objective(trial):
        model = construct_model(trial)   # sample architecture from search space
        trainer = get_trainer(model=model, tokenized_dataset=dataset, tokenizer=tokenizer,
                              evaluate_metric="accuracy", num_train_epochs=1)
        trainer.train()
        return trainer.evaluate()["eval_accuracy"]

    study = optuna.create_study(direction="maximize", sampler=RandomSampler())
    study.optimize(objective, n_trials=10)

See :doc:`tutorials/tutorial_5_nas_optuna_script` for a full walkthrough including the model
constructor and search space definition.

Model Compression
-----------------

The ``CompressionPipeline`` applies quantization and pruning in a single pass, preparing a model
for edge deployment.

.. code-block:: python

    from chop.pipelines import CompressionPipeline
    from chop import MaseGraph

    mg = MaseGraph(model)
    pipe = CompressionPipeline()

    quantization_config = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {"config": {"name": "integer",
                               "data_in_width": 8, "data_in_frac_width": 4,
                               "weight_width": 8, "weight_frac_width": 4,
                               "bias_width": 8, "bias_frac_width": 4}},
    }
    pruning_config = {
        "weight": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
        "activation": {"sparsity": 0.5, "method": "l1-norm", "scope": "local"},
    }

    mg, _ = pipe(mg, pass_args={
        "quantize_transform_pass": quantization_config,
        "prune_transform_pass": pruning_config,
    })

See :doc:`tutorials/tutorial_3_qat_script` for quantization,
:doc:`tutorials/tutorial_4_pruning_script` for pruning, and
:doc:`tutorials/tutorial_5_nas_optuna_script` for combined compression via ``CompressionPipeline``.
