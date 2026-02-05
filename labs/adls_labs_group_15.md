# ADLS Labs – Group 15

---

## Introduction

---

## Lab 0 – Introduction and Environment Setup

### Setup

### Results

### Observations

### Conclusion

---

## Lab 1 – Quantisation and Pruning Fundamentals

### Setup

### Results

### Observations

### Conclusion

---

## Lab 2 – Neural Architecture Search (NAS) and Compression-Aware Optimisation

### Overview

This lab constructs the NAS workfow for BERT, then shows how to train, evaluate and then finally compress the (best) discovered architecture. <br>
1. We start by initializing training pipeline (tokenization into input IDs and attention masks).
2. Next define a dictionary of search hyperparameters that BERT is allowed to change, such as number of transfomer layers, or even the type of layer.
3. A `construct_model` function is defined which samples the hyperparametrs from Optuna and constructs a BERT model using those settings. The evaluation of the sampling strategy from Optuna is the implementation focus of this lab.
4. An `objective` function is defined which essentially just evaluates the validation accuracy.
5. Finally, run multiple trials of each sampler and measure the accuracy across trials for each one.
6. Using the model from the sampler that yielded the best results, we finally compress the model using MASE. We may also fine-tune this model. We check how much the performance changed after the compression.

### Task 1

### Task 2
---

## Lab 3 – Mixed-Precision Quantisation Search

### Setup

### Results

### Observations

### Conclusion

### Key Takeaways


---

## Lab 4 – System Performance and torch.compile

### Setup

### Results

### Observations

### Conclusion

### Key Takeaways


---
