# Zero-Cost Proxy Evaluation Process

The zero-cost proxy evaluation process is designed to efficiently evaluate neural network (NN) architectures without the need for extensive computational resources. This process involves several steps, starting from configuration setup to the final output of NN architecture rankings and zero-cost proxy values.

## Start-up

The process begins with the setup phase, where the configuration parameters are defined in a `.toml` file.

- **Configuration (.toml)**: This file contains all necessary configurations, including the options to be explored and the zero-cost proxies to be used.

### Key Steps:

1. **Extract Key Information**: The system parses the configuration file to extract vital information, such as option choices and the zero-cost proxies that will be utilized.
2. **Construct Search Space**: Based on the extracted information, a search space is constructed, ready for exploration.

## Search Phase

The core of the evaluation process is the search phase, where different NN architectures are explored to find the most efficient ones according to the zero-cost proxies.

- **TPE Update**: The Tree-structured Parzen Estimator (TPE) model is updated with the results from evaluating the current architecture options.
- **TPE Selection**: The TPE model selects the next set of architecture options to evaluate based on the updated model.

### Detailed Steps:

1. **Zero-cost Proxies Calculation**: For each selected option combination, zero-cost proxies are calculated to evaluate the architecture without needing full training.
2. **Model Rebuild Accordingly**: The NN architecture is rebuilt based on the selected option combination to reflect the proposed changes.
3. **NN Architecture (nn.Module)**: The process considers the architecture as a module, allowing for easy manipulation and evaluation.

## Output

Once the process reaches the maximum number of iterations or completes the search space exploration, it produces the final output.

- **NN Architecture Ranking**: The architectures are ranked based on their performance as evaluated by the zero-cost proxies.
- **Zero-cost Proxy Values**: The values of the zero-cost proxies for each evaluated architecture are provided.

### Final Evaluation:

- **Evaluate Proxy Contribution**: The final step involves evaluating the contribution of each proxy to the overall performance, enabling the selection of the best scales for proxies.
