Tutorial 6: Mixed Precision Quantization Search with Mase and Optuna
=====================================================================

In this tutorial, we integrate Mase with Optuna to search for the optimal per-layer quantization
mapping for a BERT model on the IMDb sentiment classification dataset. Rather than applying uniform
integer quantization to every layer (as in Tutorial 3), we let Optuna decide — for each linear
layer independently — whether it should remain in full precision (``torch.nn.Linear``) or be
quantized to 8-bit integer (``LinearInteger``).

Running mixed-precision quantization search involves the following steps:

1. **Define the search space**: enumerate the allowed layer types for each linear layer.
2. **Write the model constructor**: a function that uses Optuna's ``suggest_categorical`` to build a
   mixed-precision model per trial.
3. **Write the objective function**: trains the sampled model for one epoch and returns accuracy.
4. **Launch the search**: choose a sampler, create a study, and run.

Run this tutorial
-----------------

From the repository root:

.. code-block:: bash

   uv run python docs/source/modules/documentation/tutorials/tutorial_6_mixed_precision_search.py

Expected terminal output (excerpt)
-----------------------------------

.. code-block:: text

   ============================================================
   Tutorial 6: Mixed Precision Search with Mase + Optuna
   ============================================================

   [1/4] Loading base model...
         Loaded from tutorial_5_best_model.pkl  ✓

   [2/4] Loading and tokenizing IMDb dataset...
   INFO     Tokenizing dataset imdb with AutoTokenizer for bert-base-uncased.
         Dataset loaded: 25000 train / 25000 test

   [3/4] Defining mixed-precision search space...
         Choices per linear layer: ['Linear', 'LinearInteger']

   [4/4] Launching mixed-precision search (1 trial(s))...
   {'loss': 0.3507, 'grad_norm': 14.654342651367188, 'learning_rate': 4.2016e-05, 'epoch': 0.16}
   ...
   {'train_runtime': 514.1645, 'train_samples_per_second': 48.623, 'train_steps_per_second': 6.078, 'train_loss': 0.3249597619628906, 'epoch': 1.0}
   Trial 1/1: 100%|██████████| 1/1 [11:06<00:00, 667.00s/trial, accuracy=0.8582]
         Best accuracy: 0.8582
         Quantized layers: 5/8

   ============================================================
   Tutorial 6 complete!
   ============================================================

Step 1: Load base model
-----------------------

This tutorial builds on Tutorial 5. The default option loads the best NAS model saved at the end
of Tutorial 5.

.. literalinclude:: tutorial_6_mixed_precision_search.py
   :language: python
   :start-after: # ── Step 1: Load base model ────────────────────────────────────────────────────
   :end-before: # ── Step 2: Load dataset ───────────────────────────────────────────────────────

Example output:

.. code-block:: text

   [1/4] Loading base model...
         Loaded from tutorial_5_best_model.pkl  ✓

If Tutorial 5 has not been run yet, you can load the model directly from HuggingFace instead
(comment out Option A and uncomment Option B in the script):

.. code-block:: python

   base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

Step 2: Load dataset
--------------------

.. literalinclude:: tutorial_6_mixed_precision_search.py
   :language: python
   :start-after: # ── Step 2: Load dataset ───────────────────────────────────────────────────────
   :end-before: # ── Step 3: Define search space & model constructor ───────────────────────────

Example output:

.. code-block:: text

   [2/4] Loading and tokenizing IMDb dataset...
   INFO     Tokenizing dataset imdb with AutoTokenizer for bert-base-uncased.
         Dataset loaded: 25000 train / 25000 test

1. Defining the Search Space
-----------------------------

Unlike Tutorial 5 (NAS), the search space here is over **quantization format** rather than
architecture. For each linear layer in the model, Optuna can choose between:

- ``torch.nn.Linear``: full-precision floating-point layer (unchanged).
- ``LinearInteger``: 8-bit integer quantized layer (``data_in_width=8``, ``weight_width=8``).

Many other quantized formats are available in ``chop.nn.quantized.modules.linear`` — such as
``LinearMinifloatIEEE``, ``LinearLog``, ``LinearBlockFP``, ``LinearBinary`` — and can be added
to the search space for richer exploration.

2. Writing a Model Constructor
--------------------------------

``construct_model`` deep-copies the base model and iterates over all linear layers. For each one,
it calls ``trial.suggest_categorical`` to choose the layer type. If ``LinearInteger`` is chosen,
the original weight data is copied into the new quantized layer.

3. Defining the Objective Function
------------------------------------

The objective trains the mixed-precision model for one epoch with ``get_trainer`` and reports
accuracy on the test split.

Step 3: Define search space and model constructor
--------------------------------------------------

.. literalinclude:: tutorial_6_mixed_precision_search.py
   :language: python
   :start-after: # ── Step 3: Define search space & model constructor ───────────────────────────
   :end-before: # ── Step 4: Launch mixed-precision search ─────────────────────────────────────

Step 4: Launch mixed-precision search
--------------------------------------

.. literalinclude:: tutorial_6_mixed_precision_search.py
   :language: python
   :start-after: # ── Step 4: Launch mixed-precision search ─────────────────────────────────────

Example output:

.. code-block:: text

   [4/4] Launching mixed-precision search (1 trial(s))...
   {'loss': 0.3507, 'grad_norm': 14.654342651367188, 'learning_rate': 4.2016e-05, 'epoch': 0.16}
   {'loss': 0.3085, 'grad_norm': 25.187883377075195, 'learning_rate': 3.4016e-05, 'epoch': 0.32}
   {'loss': 0.3084, 'grad_norm': 33.81252670288086, 'learning_rate': 2.6016000000000003e-05, 'epoch': 0.48}
   {'loss': 0.3211, 'grad_norm': 57.05055236816406, 'learning_rate': 1.8015999999999998e-05, 'epoch': 0.64}
   {'loss': 0.3281, 'grad_norm': 20.263505935668945, 'learning_rate': 1.0016e-05, 'epoch': 0.8}
   {'loss': 0.3311, 'grad_norm': 46.44586181640625, 'learning_rate': 2.0160000000000003e-06, 'epoch': 0.96}
   {'train_runtime': 514.1645, 'train_samples_per_second': 48.623, 'train_steps_per_second': 6.078, 'train_loss': 0.3249597619628906, 'epoch': 1.0}
   Trial 1/1: 100%|██████████| 1/1 [11:06<00:00, 667.00s/trial, accuracy=0.8582]
         Best accuracy: 0.8582
         Quantized layers: 5/8

Conclusion
----------

Tutorial 6 demonstrates per-layer mixed-precision quantization search:

- With ``N_TRIALS=1`` the result is a single random assignment of layer types; the search found
  that quantizing 5 out of 8 linear layers achieves **0.8582** accuracy.
- Increase ``N_TRIALS`` (e.g., 50–100) to explore the trade-off between quantized layers and
  accuracy more thoroughly.
- The search space can be extended with additional quantization formats from
  ``chop.nn.quantized.modules.linear`` for finer-grained mixed-precision mapping.

**Task**: Add ``LinearMinifloatIEEE`` to ``search_space["linear_layer_choices"]`` and rerun the
search. Compare the best accuracy and the proportion of quantized layers against the
``LinearInteger``-only run.
