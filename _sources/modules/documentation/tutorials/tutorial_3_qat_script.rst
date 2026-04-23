Tutorial 3: Running Quantization-Aware Training (QAT) on Bert
===============================================================

In this tutorial, we build on top of Tutorial 2 by taking a BERT sequence-classification model and running
MASE quantization passes. We first run Post-Training Quantization (PTQ), then continue training with
Quantization-Aware Training (QAT) to recover quantized-model accuracy.

Run this tutorial
-----------------

From the repository root:

.. code-block:: bash

   uv run python docs/source/modules/documentation/tutorials/tutorial_3_qat.py

Expected terminal output (excerpt)
----------------------------------

.. code-block:: text

   ============================================================
   Tutorial 3: Post-Training Quantization + QAT on BERT
   ============================================================

   [1/5] Loading model...
   WARNING  Node finfo not found in loaded metadata.
   WARNING  Node getattr_2 not found in loaded metadata.
         Loaded from tutorial_2_lora  ✓

   [2/5] Loading dataset and evaluating baseline accuracy...
   INFO     Tokenizing dataset imdb with AutoTokenizer for bert-base-uncased.
         Dataset loaded: 25000 train / 25000 test
         [Baseline] Accuracy: 0.8350

   [3/5] Applying integer quantization (PTQ)...
         quantize_transform_pass  ✓
         [PTQ] Accuracy: 0.7738
   INFO     Exporting MaseGraph to ~/tutorial_3_ptq.pt, ~/tutorial_3_ptq.mz
   INFO     Exporting GraphModule to ~/tutorial_3_ptq.pt
   INFO     Saving full model format
   INFO     Exporting MaseMetadata to ~/tutorial_3_ptq.mz
         PTQ checkpoint saved to ~/tutorial_3_ptq

   [4/5] Running QAT (1 epoch)...
   {'loss': 0.4101, 'grad_norm': 10.378958702087402, 'learning_rate': 4.2016e-05, 'epoch': 0.16}
   {'loss': 0.401, 'grad_norm': 5.116357326507568, 'learning_rate': 3.4016e-05, 'epoch': 0.32}
   {'loss': 0.3959, 'grad_norm': 15.316791534423828, 'learning_rate': 2.6016000000000003e-05, 'epoch': 0.48}
   ...
   {'train_runtime': 141.1579, 'train_samples_per_second': 177.107, 'train_steps_per_second': 22.138, 'train_loss': 0.39675407958984377, 'epoch': 1.0}
         [QAT] Accuracy: 0.8399

   [5/5] Exporting QAT checkpoint...
   INFO     Exporting MaseGraph to ~/tutorial_3_qat.pt, ~/tutorial_3_qat.mz
   INFO     Exporting GraphModule to ~/tutorial_3_qat.pt
   INFO     Saving full model format
   INFO     Exporting MaseMetadata to ~/tutorial_3_qat.mz
         QAT checkpoint saved to ~/tutorial_3_qat

   ============================================================
   Tutorial 3 complete!
   ============================================================

Importing the model
-------------------

If you are starting from scratch, create a fresh MaseGraph for BERT:

Step 1: Import model and build MaseGraph
-----------------------------------------

.. literalinclude:: tutorial_3_qat.py
   :language: python
   :start-after: # ── Step 1: Import model ───────────────────────────────────────────────────────
   :end-before: # ── Step 2: Baseline evaluation ────────────────────────────────────────────────

Example output:

.. code-block:: text

   [1/5] Loading model...
   WARNING  Node finfo not found in loaded metadata.
   WARNING  Node getattr_2 not found in loaded metadata.
         Loaded from tutorial_2_lora  ✓

If Tutorial 2 has not been run yet, you can build a fresh MaseGraph instead (comment out Option A and uncomment Option B in the script):

.. code-block:: python

   model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
   model.config.problem_type = "single_label_classification"
   mg = MaseGraph(model, hf_input_names=["input_ids", "attention_mask", "labels"])
   mg, _ = passes.init_metadata_analysis_pass(mg)
   mg, _ = passes.add_common_metadata_analysis_pass(mg)

Post-Training Quantization (PTQ)
--------------------------------

Before quantization, evaluate baseline accuracy with the tokenized IMDb dataset and HuggingFace trainer.

Step 2: Baseline evaluation
---------------------------

.. literalinclude:: tutorial_3_qat.py
   :language: python
   :start-after: # ── Step 2: Baseline evaluation ────────────────────────────────────────────────
   :end-before: # ── Step 3: Post-Training Quantization (PTQ) ──────────────────────────────────

Example output:

.. code-block:: text

   [2/5] Loading dataset and evaluating baseline accuracy...
   INFO     Tokenizing dataset imdb with AutoTokenizer for bert-base-uncased.
         Dataset loaded: 25000 train / 25000 test
         [Baseline] Accuracy: 0.8350

Next, run quantization with a "by type" config, where quantization is assigned by ``mase_op``. In this
tutorial, all linear activations/weights/biases are quantized with the same integer precision.

Step 3: Apply PTQ and evaluate
------------------------------

.. literalinclude:: tutorial_3_qat.py
   :language: python
   :start-after: # ── Step 3: Post-Training Quantization (PTQ) ──────────────────────────────────
   :end-before: # ── Step 4: Quantization-Aware Training (QAT) ─────────────────────────────────

Example output:

.. code-block:: text

   [3/5] Applying integer quantization (PTQ)...
         quantize_transform_pass  ✓
         [PTQ] Accuracy: 0.7738
   INFO     Exporting MaseGraph to ~/tutorial_3_ptq.pt, ~/tutorial_3_ptq.mz
   INFO     Exporting GraphModule to ~/tutorial_3_ptq.pt
   INFO     Saving full model format
   INFO     Exporting MaseMetadata to ~/tutorial_3_ptq.mz
         PTQ checkpoint saved to ~/tutorial_3_ptq

Quantization-Aware Training (QAT)
---------------------------------

PTQ alone can reduce accuracy. To reduce this performance gap, include the quantized model back in the
training loop and fine-tune with QAT.

Step 4: Run QAT
---------------

.. literalinclude:: tutorial_3_qat.py
   :language: python
   :start-after: # ── Step 4: Quantization-Aware Training (QAT) ─────────────────────────────────
   :end-before: # ── Step 5: Export QAT checkpoint ─────────────────────────────────────────────

Example output:

.. code-block:: text

   {'loss': 0.4101, 'grad_norm': 10.378958702087402, 'learning_rate': 4.2016e-05, 'epoch': 0.16}
   {'loss': 0.401, 'grad_norm': 5.116357326507568, 'learning_rate': 3.4016e-05, 'epoch': 0.32}
   {'loss': 0.3959, 'grad_norm': 15.316791534423828, 'learning_rate': 2.6016000000000003e-05, 'epoch': 0.48}
   {'loss': 0.3906, 'grad_norm': 9.798357009887695, 'learning_rate': 1.8015999999999998e-05, 'epoch': 0.64}
   {'loss': 0.3874, 'grad_norm': 6.183642864227295, 'learning_rate': 1.0016e-05, 'epoch': 0.8}
   {'loss': 0.3918, 'grad_norm': 10.731794357299805, 'learning_rate': 2.0160000000000003e-06, 'epoch': 0.96}
   {'train_runtime': 141.1579, 'train_samples_per_second': 177.107, 'train_steps_per_second': 22.138, 'train_loss': 0.39675407958984377, 'epoch': 1.0}
         [QAT] Accuracy: 0.8399

Step 5: Export final QAT checkpoint
-----------------------------------

.. literalinclude:: tutorial_3_qat.py
   :language: python
   :start-after: # ── Step 5: Export QAT checkpoint ─────────────────────────────────────────────

Example output:

.. code-block:: text

   INFO     Exporting MaseGraph to ~/tutorial_3_qat.pt, ~/tutorial_3_qat.mz
   INFO     Exporting GraphModule to ~/tutorial_3_qat.pt
   INFO     Saving full model format
   INFO     Exporting MaseMetadata to ~/tutorial_3_qat.mz
         QAT checkpoint saved to ~/tutorial_3_qat

Conclusion
----------

Tutorial 3 demonstrates a standard PTQ→QAT workflow:

- PTQ gives a quick quantized baseline and can reduce model accuracy.
- QAT can recover (and in some runs exceed) full-precision accuracy.
- Exported checkpoints are saved for follow-up tutorials:

  - ``tutorial_3_ptq.pt`` and ``tutorial_3_ptq.mz``
  - ``tutorial_3_qat.pt`` and ``tutorial_3_qat.mz``

