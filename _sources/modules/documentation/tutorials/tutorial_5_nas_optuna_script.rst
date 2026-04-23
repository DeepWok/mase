Tutorial 5: Neural Architecture Search (NAS) with Mase and Optuna
==================================================================

In this tutorial, we'll see how Mase can be integrated with Optuna, the popular hyperparameter
optimization framework, to search for a BERT model optimized for sequence classification on the
IMDb dataset. We'll take the Optuna-generated model and import it into Mase, then run the
``CompressionPipeline`` to prepare the model for edge deployment by quantizing and pruning its weights.

Running Architecture Search with Mase/Optuna involves the following steps:

1. **Define the search space**: a dictionary containing the range of values for each parameter at each layer.
2. **Write the model constructor**: a function that uses Optuna utilities to sample a model from the search space.
3. **Write the objective function**: calls the model constructor and defines training/evaluation for each trial.
4. **Launch the search**: choose an Optuna sampler, create a study, and run.

Run this tutorial
-----------------

From the repository root:

.. code-block:: bash

   uv run python docs/source/modules/documentation/tutorials/tutorial_5_nas_optuna.py

Expected terminal output (excerpt)
-----------------------------------

.. code-block:: text

   ============================================================
   Tutorial 5: NAS with Mase + Optuna (BERT / IMDb)
   ============================================================

   [1/5] Loading and tokenizing IMDb dataset...
   INFO     Tokenizing dataset imdb with AutoTokenizer for bert-base-uncased.
         Dataset loaded: 25000 train / 25000 test

   [2/5] Defining NAS search space...
         num_layers: [2, 4, 8]
         num_heads: [2, 4, 8, 16]
         hidden_size: [128, 192, 256, 384, 512]
         intermediate_size: [512, 768, 1024, 1536, 2048]
         linear_layer_choices: [<class 'torch.nn.modules.linear.Linear'>, <class 'chop.nn.modules.identity.Identity'>]

   [3/5] Launching NAS search (1 trial(s))...
   Trial 1/1: 100%|██████████| 1/1 [10:06<00:00, 606.87s/trial, accuracy=0.8616]
         Best accuracy: 0.8616
         Best params:   {'num_layers': 1, 'num_heads': 1, 'hidden_size': 4, 'intermediate_size': 3, ...}

   [4/5] Saving best model...
         Saved to ~/tutorial_5_best_model.pkl

   [5/5] Compressing best model (quantize + prune)...
   INFO     Pruning module: bert_encoder_layer_0_attention_self_value
   INFO     Pruning module: bert_encoder_layer_0_attention_output_dense
   ...
   INFO     Pruning module: classifier
   INFO     Exporting MaseGraph to ~/tutorial_5_nas_compressed.pt, ~/tutorial_5_nas_compressed.mz
   INFO     Saving state_dict format
         Compressed model saved to ~/tutorial_5_nas_compressed

   ============================================================
   Tutorial 5 complete!
   ============================================================

Step 1: Load dataset
--------------------

.. literalinclude:: tutorial_5_nas_optuna.py
   :language: python
   :start-after: # ── Step 1: Load dataset ───────────────────────────────────────────────────────
   :end-before: # ── Step 2: Define search space ───────────────────────────────────────────────

Example output:

.. code-block:: text

   [1/5] Loading and tokenizing IMDb dataset...
   INFO     Tokenizing dataset imdb with AutoTokenizer for bert-base-uncased.
         Dataset loaded: 25000 train / 25000 test

1. Defining the Search Space
-----------------------------

We enumerate the possible combinations of hyperparameters that Optuna can choose during search.
We explore the following range of values for hidden size, intermediate size, number of layers and
number of heads, inspired by the `NAS-BERT paper <https://arxiv.org/abs/2105.14444>`_.

Additionally, for each linear layer where ``in_features == out_features``, we allow Optuna to
replace it with an ``Identity`` module (effectively skipping that layer).

Step 2: Define search space
----------------------------

.. literalinclude:: tutorial_5_nas_optuna.py
   :language: python
   :start-after: # ── Step 2: Define search space ───────────────────────────────────────────────
   :end-before: # ── Step 3: Model constructor + objective ─────────────────────────────────────

Example output:

.. code-block:: text

   [2/5] Defining NAS search space...
         num_layers: [2, 4, 8]
         num_heads: [2, 4, 8, 16]
         hidden_size: [128, 192, 256, 384, 512]
         intermediate_size: [512, 768, 1024, 1536, 2048]
         linear_layer_choices: [<class 'torch.nn.modules.linear.Linear'>, <class 'chop.nn.modules.identity.Identity'>]

2. Writing a Model Constructor
--------------------------------

We define ``construct_model``, which is called on each trial. The function receives the ``trial``
argument — an Optuna object with ``suggest_int`` and ``suggest_categorical`` methods that trigger
the chosen sampler to pick parameter values.

3. Defining the Objective Function
------------------------------------

The objective function calls ``construct_model``, trains for one epoch using ``get_trainer``, and
reports classification accuracy on the test split. The model is also stored as a trial attribute
so we can retrieve it after the search.

Step 3: Model constructor and objective function
-------------------------------------------------

.. literalinclude:: tutorial_5_nas_optuna.py
   :language: python
   :start-after: # ── Step 3: Model constructor + objective ─────────────────────────────────────
   :end-before: # ── Step 4: Launch NAS search ─────────────────────────────────────────────────

4. Launching the Search
------------------------

Optuna provides a number of samplers:

- **GridSampler**: iterates through every possible combination of hyperparameters.
- **RandomSampler**: chooses a random combination in each iteration.
- **TPESampler**: uses Tree-structured Parzen Estimator to choose hyperparameter values.

The number of trials is set to 1 so each run completes in about 10 minutes. For better results,
set ``N_TRIALS`` to 100 and leave it running overnight.

Step 4: Launch NAS search
--------------------------

.. literalinclude:: tutorial_5_nas_optuna.py
   :language: python
   :start-after: # ── Step 4: Launch NAS search ─────────────────────────────────────────────────
   :end-before: # ── Step 5: Save best model ────────────────────────────────────────────────────

Example output:

.. code-block:: text

   [3/5] Launching NAS search (1 trial(s))...
   {'loss': 0.7074, 'grad_norm': 2.403750419616699, 'learning_rate': 4.2016e-05, 'epoch': 0.16}
   {'loss': 0.5646, 'grad_norm': 10.625929832458496, 'learning_rate': 3.4016e-05, 'epoch': 0.32}
   {'loss': 0.4476, 'grad_norm': 33.16868591308594, 'learning_rate': 2.6016000000000003e-05, 'epoch': 0.48}
   {'loss': 0.3991, 'grad_norm': 41.173095703125, 'learning_rate': 1.8015999999999998e-05, 'epoch': 0.64}
   {'loss': 0.371, 'grad_norm': 21.3480224609375, 'learning_rate': 1.0016e-05, 'epoch': 0.8}
   {'loss': 0.3499, 'grad_norm': 23.04668617248535, 'learning_rate': 2.0160000000000003e-06, 'epoch': 0.96}
   {'train_runtime': 453.3577, 'train_samples_per_second': 55.144, 'train_steps_per_second': 6.893, 'train_loss': 0.4683348876953125, 'epoch': 1.0}
   Trial 1/1: 100%|██████████| 1/1 [10:06<00:00, 606.87s/trial, accuracy=0.8616]
         Best accuracy: 0.8616
         Best params:   {'num_layers': 1, 'num_heads': 1, 'hidden_size': 4, 'intermediate_size': 3,
                          'bert.encoder.layer.0.attention.self.query_type': Identity,
                          'bert.encoder.layer.0.attention.self.key_type': Identity,
                          'bert.encoder.layer.0.attention.self.value_type': Linear,
                          'bert.encoder.layer.0.attention.output.dense_type': Linear,
                          ...}

Step 5: Save best model
------------------------

After the search, we retrieve the best trial's model and serialize it with ``dill`` for use in
Tutorial 6 (mixed-precision search).

.. literalinclude:: tutorial_5_nas_optuna.py
   :language: python
   :start-after: # ── Step 5: Save best model ────────────────────────────────────────────────────
   :end-before: # ── Step 6: Compress with CompressionPipeline ─────────────────────────────────

Example output:

.. code-block:: text

   [4/5] Saving best model...
         Saved to ~/tutorial_5_best_model.pkl

Deploying the Optimized Model with CompressionPipeline
-------------------------------------------------------

Now we run the ``CompressionPipeline`` in Mase to apply uniform quantization and pruning over the
searched model in a single pass.

Step 6: Compress with CompressionPipeline
------------------------------------------

.. literalinclude:: tutorial_5_nas_optuna.py
   :language: python
   :start-after: # ── Step 6: Compress with CompressionPipeline ─────────────────────────────────

Example output:

.. code-block:: text

   [5/5] Compressing best model (quantize + prune)...
   INFO     Getting dummy input for prajjwal1/bert-tiny.
   INFO     Pruning module: bert_encoder_layer_0_attention_self_value
   INFO     Pruning module: bert_encoder_layer_0_attention_output_dense
   INFO     Pruning module: bert_encoder_layer_0_intermediate_dense
   INFO     Pruning module: bert_encoder_layer_0_output_dense
   INFO     Pruning module: bert_encoder_layer_1_attention_output_dense
   INFO     Pruning module: bert_encoder_layer_1_intermediate_dense
   INFO     Pruning module: bert_encoder_layer_1_output_dense
   INFO     Pruning module: classifier
   INFO     Exporting MaseGraph to ~/tutorial_5_nas_compressed.pt, ~/tutorial_5_nas_compressed.mz
   INFO     Exporting GraphModule to ~/tutorial_5_nas_compressed.pt
   INFO     Saving state_dict format
   INFO     Exporting MaseMetadata to ~/tutorial_5_nas_compressed.mz
   WARNING  Failed to pickle call_function node: finfo
   WARNING  cannot pickle 'torch.finfo' object
         Compressed model saved to ~/tutorial_5_nas_compressed

.. note::

   The ``WARNING: Failed to pickle call_function node: finfo`` messages are expected when exporting
   models containing ``torch.finfo`` calls. The checkpoint is still saved successfully.

Conclusion
----------

Tutorial 5 demonstrates a full NAS + compression workflow:

- The search space covers architecture dimensions (layers, heads, hidden size) and layer type
  (``Linear`` vs ``Identity``).
- With ``N_TRIALS=1`` the result is essentially a random architecture; increase trials for a
  meaningful search.
- The best model is serialized to ``~/tutorial_5_best_model.pkl`` for use in Tutorial 6.
- The ``CompressionPipeline`` applies quantization and pruning in a single pass, producing a
  compressed checkpoint at ``~/tutorial_5_nas_compressed``.

**Task**: Increase ``N_TRIALS`` to 10 (or more) and compare the best accuracy across runs. You
should observe that more trials improve the best found architecture.
