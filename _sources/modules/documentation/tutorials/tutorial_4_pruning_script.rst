Tutorial 4: Unstructured Pruning on Bert
=========================================

Pruning is a technique used to reduce the size and complexity of neural networks by removing unnecessary
parameters (weights and connections). The goal is to create a smaller, more efficient model that maintains
most of the original model's performance.

The following benefits can be seen from pruning neural networks:

- **Reduce model size**: Deep neural networks often have millions of parameters, leading to large storage requirements.
- **Decrease inference time**: Fewer parameters mean fewer computations, resulting in faster predictions.
- **Improve generalization**: Removing unnecessary connections can help prevent overfitting.
- **Energy efficiency**: Smaller models require less energy to run, which is crucial for edge devices and mobile applications.

Structured pruning removes entire structures (e.g., channels, filters, or layers) from the network, while
**unstructured pruning** removes individual weights or connections regardless of their location. In this
tutorial, we build on top of Tutorial 3 by taking the quantized BERT model and running MASE's unstructured
pruning pass. After pruning, we run further fine-tuning iterations to retain sequence classification accuracy.

Run this tutorial
-----------------

From the repository root:

.. code-block:: bash

   uv run python docs/source/modules/documentation/tutorials/tutorial_4_pruning.py

Expected terminal output (excerpt)
-----------------------------------

.. code-block:: text

   ============================================================
   Tutorial 4: Unstructured Pruning on BERT (IMDb)
   ============================================================

   [1/5] Loading model...
         Loaded from tutorial_3_qat  ✓

   [2/5] Loading dataset and evaluating baseline accuracy...
   INFO     Tokenizing dataset imdb with AutoTokenizer for bert-base-uncased.
         Dataset loaded: 25000 train / 25000 test
         [Baseline] Accuracy: 0.8399

   [3/5] Applying L1-norm unstructured pruning (sparsity=0.5)...
   INFO     Pruning module: bert_encoder_layer_0_attention_self_query
   INFO     Pruning module: bert_encoder_layer_0_attention_self_key
   ...
   INFO     Pruning module: classifier
         prune_transform_pass  ✓

   [4/5] Evaluating accuracy after pruning (before finetuning)...
         [Pruned] Accuracy: 0.7284

   [5/5] Finetuning pruned model (5 epochs) to recover accuracy...
   {'loss': 0.4703, 'grad_norm': 1.3808, 'learning_rate': 4.8403e-05, 'epoch': 0.16}
   ...
   {'train_runtime': 1321.7466, 'train_samples_per_second': 94.572, 'train_steps_per_second': 11.821, 'train_loss': 0.39856, 'epoch': 5.0}
         [Pruned + finetuned] Accuracy: 0.8311

   ============================================================
   Tutorial 4 complete!
   ============================================================

Importing the model
-------------------

This tutorial builds on Tutorial 3. The default option loads the QAT checkpoint saved at the end of
Tutorial 3.

Step 1: Import model
---------------------

.. literalinclude:: tutorial_4_pruning.py
   :language: python
   :start-after: # ── Step 1: Import model ───────────────────────────────────────────────────────
   :end-before: # ── Step 2: Load dataset & baseline evaluation ────────────────────────────────

Example output:

.. code-block:: text

   [1/5] Loading model...
         Loaded from tutorial_3_qat  ✓

If Tutorial 3 has not been run yet, you can build a fresh MaseGraph instead (comment out Option A and
uncomment Option B in the script):

.. code-block:: python

   model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
   model.config.problem_type = "single_label_classification"
   mg = MaseGraph(model, hf_input_names=["input_ids", "attention_mask", "labels"])
   mg, _ = passes.init_metadata_analysis_pass(mg)
   mg, _ = passes.add_common_metadata_analysis_pass(mg)

Unstructured Pruning
--------------------

Before running pruning, we evaluate the model accuracy on the IMDb dataset. If you are coming from
Tutorial 3, this should be the same as the accuracy after Quantization-Aware Training (QAT). If
you have just initialized the model, this will likely be around 50% (random guess), in which case
pruning would not have a significant effect on accuracy.

Step 2: Load dataset and baseline evaluation
---------------------------------------------

.. literalinclude:: tutorial_4_pruning.py
   :language: python
   :start-after: # ── Step 2: Load dataset & baseline evaluation ────────────────────────────────
   :end-before: # ── Step 3: Apply L1-norm unstructured pruning (50% sparsity) ─────────────────

Example output:

.. code-block:: text

   [2/5] Loading dataset and evaluating baseline accuracy...
   INFO     Tokenizing dataset imdb with AutoTokenizer for bert-base-uncased.
         Dataset loaded: 25000 train / 25000 test
         [Baseline] Accuracy: 0.8399

To run the pruning pass, we pass the following pruning configuration dictionary, which defines:

- **Sparsity**: a value between 0 and 1, expressing the proportion of elements to prune (set to 0).
- **Method**: pruning methods supported include ``random`` and ``l1-norm``.
- **Scope**: whether to consider each tensor individually (``local``) or all tensors together (``global``) when computing pruning statistics.

We use L1-norm pruning with local scope at 50% sparsity. This removes the weights with the smallest
absolute values within each layer independently.

Step 3: Apply L1-norm unstructured pruning
-------------------------------------------

.. literalinclude:: tutorial_4_pruning.py
   :language: python
   :start-after: # ── Step 3: Apply L1-norm unstructured pruning (50% sparsity) ─────────────────
   :end-before: # ── Step 4: Evaluate after pruning ────────────────────────────────────────────

Example output:

.. code-block:: text

   [3/5] Applying L1-norm unstructured pruning (sparsity=0.5)...
   INFO     Pruning module: bert_encoder_layer_0_attention_self_query
   INFO     Pruning module: bert_encoder_layer_0_attention_self_key
   INFO     Pruning module: bert_encoder_layer_0_attention_self_value
   INFO     Pruning module: bert_encoder_layer_0_attention_output_dense
   INFO     Pruning module: bert_encoder_layer_0_intermediate_dense
   INFO     Pruning module: bert_encoder_layer_0_output_dense
   INFO     Pruning module: bert_encoder_layer_1_attention_self_query
   INFO     Pruning module: bert_encoder_layer_1_attention_self_key
   INFO     Pruning module: bert_encoder_layer_1_attention_self_value
   INFO     Pruning module: bert_encoder_layer_1_attention_output_dense
   INFO     Pruning module: bert_encoder_layer_1_intermediate_dense
   INFO     Pruning module: bert_encoder_layer_1_output_dense
   INFO     Pruning module: bert_pooler_dense
   INFO     Pruning module: classifier
         prune_transform_pass  ✓

Step 4: Evaluate accuracy after pruning
-----------------------------------------

Let's evaluate the immediate effect of pruning on accuracy. It is likely to observe drops of around
10% or more.

.. literalinclude:: tutorial_4_pruning.py
   :language: python
   :start-after: # ── Step 4: Evaluate after pruning ────────────────────────────────────────────
   :end-before: # ── Step 5: Finetune to recover accuracy ──────────────────────────────────────

Example output:

.. code-block:: text

   [4/5] Evaluating accuracy after pruning (before finetuning)...
         [Pruned] Accuracy: 0.7284

Step 5: Finetune to recover accuracy
--------------------------------------

To overcome the drop in accuracy, we run finetuning epochs on the pruned model. This allows the model
to adapt to the new pruning mask.

.. literalinclude:: tutorial_4_pruning.py
   :language: python
   :start-after: # ── Step 5: Finetune to recover accuracy ──────────────────────────────────────

Example output:

.. code-block:: text

   [5/5] Finetuning pruned model (5 epochs) to recover accuracy...
   {'loss': 0.4703, 'grad_norm': 1.380826473236084, 'learning_rate': 4.8403200000000004e-05, 'epoch': 0.16}
   {'loss': 0.4248, 'grad_norm': 0.7311906814575195, 'learning_rate': 4.68032e-05, 'epoch': 0.32}
   {'loss': 0.4266, 'grad_norm': 1.6150014400482178, 'learning_rate': 4.52032e-05, 'epoch': 0.48}
   {'loss': 0.4096, 'grad_norm': 1.2620729207992554, 'learning_rate': 4.36032e-05, 'epoch': 0.64}
   {'loss': 0.4126, 'grad_norm': 0.5686410665512085, 'learning_rate': 4.2003200000000005e-05, 'epoch': 0.8}
   {'loss': 0.4122, 'grad_norm': 1.589716911315918, 'learning_rate': 4.04032e-05, 'epoch': 0.96}
   ...
   {'loss': 0.3835, 'grad_norm': 0.9885391592979431, 'learning_rate': 4.032e-07, 'epoch': 4.96}
   {'train_runtime': 1321.7466, 'train_samples_per_second': 94.572, 'train_steps_per_second': 11.821, 'train_loss': 0.39856319970703125, 'epoch': 5.0}
         [Pruned + finetuned] Accuracy: 0.8311

Conclusion
----------

Tutorial 4 demonstrates a standard pruning + finetuning workflow:

- Unstructured L1-norm pruning at 50% sparsity causes an accuracy drop from ~0.84 to ~0.73.
- Five epochs of finetuning recovers the accuracy back to ~0.83, close to the pre-pruning baseline.
- Pruning and quantization (from Tutorial 3) can be combined to achieve both weight compression and
  reduced numerical precision simultaneously.

**Task**: Try changing the ``sparsity`` value (e.g., ``0.3`` or ``0.7``) and observe how the pruned
accuracy and finetuning recovery change.
