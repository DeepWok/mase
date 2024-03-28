# Calculation Formulas for Zero-Cost Proxy Metrics

In our project, we employ several zero-cost proxy metrics to evaluate neural network architectures without extensive training. Below are detailed explanations and the mathematical formulas for each of these metrics:

## Proxy Metrics:

### Grad-norm (Gradient Norm)

- **Description**: The L2 norm of the gradient of the loss function with respect to the model parameters. It measures the magnitude of the gradient, providing insight into the learning capacity of the network.
- **Formula**:  $\text{Grad-norm} = \| \nabla_{\theta} L \|$
  where \(L\) is the loss function and \(\theta\) represents the model parameters.

### Fisher Information Matrix (FIM)

- **Description**: Represents the expectation of the outer product of the gradient of the log likelihood with respect to the parameters. It is used to estimate the amount of information that the observable data X carries about the unknown parameters of a model.
- **Formula**:  $F_{ij} = E\left[\frac{\partial \log p(x;\theta)}{\partial \theta_i} \frac{\partial \log p(x;\theta)}{\partial \theta_j}\right]$

### L2 Norm (of the model parameters)

- **Description**: The square root of the sum of the squared values of the parameters. It provides a measure of the magnitude of the parameters.
- **Formula**:  $\text{L2\_norm} = \sqrt{\sum_{i} \theta_i^2}$

### NASWOT\_ReLU

- **Description**: Counts the ReLU activations in the model, reflecting the network's non-linear capacity.
- **Metric**: Structural count based on the architecture's design.

### NASWOT\_Conv

- **Description**: Counts the convolutional layers within the neural network, indicating the depth and complexity of the network.
- **Metric**: Structural count based on the architecture's design.

### Plain

- **Description**: Typically refers to the accuracy metric, representing the proportion of correct predictions.
- **Formula**:  $\text{Plain} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$

### SNIP

- **Description**: Relates to the sensitivity of the network to parameter pruning, which can indicate how changes in the architecture affect performance.
- **Formula**:  $ \text{SNIP} = \left|\frac{\partial L}{\partial \theta}\right| \times |\theta|$

### SynFlow

- **Description**: Evaluates the flow of gradients through the network without training, indicating potential learning pathways.
- **Formula**:  $\text{SynFlow} = \sum_{\theta} \left|\theta \frac{\partial L}{\partial \theta}\right|$

### TENAS

- **Description**: Evaluates the sensitivity of the model to the input features, tailored for NAS. It aids in identifying architectures more suited to the data.
- **Metric**: Derived from the NAS optimization process.

### ZICO

- **Description**: Measures sensitivity to input features, used in different contexts compared to SynFlow.
- **Metric**: Varies based on implementation, typically related to input feature sensitivity.
