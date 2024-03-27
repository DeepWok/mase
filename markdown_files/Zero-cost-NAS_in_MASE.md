# Zero-Cost NAS Strategy in MASE

This document outlines the integration of NAS-Bench architectures into the MASE framework and the application of a zero-cost search strategy to efficiently evaluate and select optimal neural network architectures.

## Integration of NAS-Bench Architectures

The process begins with the definition of a search space in `graph.py`, specifically tailored for mixed-precision post-training quantization on the MASE graph. The search space utilizes NAS-Bench-201 API to query performance metrics of predefined architectures, serving as a base for further operations.

### Key Steps:

1. **Initialization**: Upon the instantiation of the `ZeroCostProxy` class, default configurations are set up, pulling an architecture from NAS-Bench-201 as a starting point.
2. **Model Rebuilding**: The `rebuild_model` method dynamically rebuilds the model based on sampled configurations. This method utilizes the NAS-Bench-201 API to extract an architecture's performance data for further analysis.
3. **Search Space Construction**: The `build_search_space` method outlines the available choices for architecture components, facilitating the exploration of various configurations.

## Zero-Cost Search Strategy

The zero-cost search strategy aims to evaluate and optimize neural network architectures without extensive computational resources. This approach significantly reduces the search time by relying on proxy metrics rather than full-fledged training.

### Implementation in `search.py`:

1. **Configuration Parsing**: The search configuration, including strategy and search space, is parsed and sanity-checked at the beginning of the search process.
2. **Search Space and Strategy Setup**: The search space is built using the `ZeroCostProxy` class, and a corresponding search strategy is instantiated. This setup allows for a structured exploration of architecture configurations.
3. **Execution of Zero-Cost Strategy**: If enabled, the zero-cost mode within the strategy class activates. This mode involves:

   - The calculation of weights for proxy metrics based on their correlation with actual performance metrics.
   - Logging of proxy metrics and predicted accuracies for the evaluated architectures.
   - Saving the evaluation results for analysis and further use.

### Results and Analysis:

The search procedure concludes with the ranking of architectures based on predicted accuracies from zero-cost proxies. These results are saved and can be analyzed to identify the most efficient architectures for specific tasks.

## Conclusion

The integration of NAS-Bench architectures into MASE, coupled with a zero-cost search strategy, presents a powerful approach to neural network architecture optimization. This methodology allows for rapid exploration and evaluation of architectures, paving the way for efficient model selection and deployment.
