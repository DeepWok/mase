import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Corrected results structure
results = [
    {
        "sparsity": 0.1,
        "l1_initial_pruned_accuracy": 0.84368,
        "l1_fine_tuned_accuracy": 0.8604,
        "rand_initial_pruned_accuracy": 0.73676,
        "rand_fine_tuned_accuracy": 0.83952
    },
    {
        "sparsity": 0.2,
        "l1_initial_pruned_accuracy": 0.83992,
        "l1_fine_tuned_accuracy": 0.85832,
        "rand_initial_pruned_accuracy": 0.6308,
        "rand_fine_tuned_accuracy": 0.8232
    },
    {
        "sparsity": 0.3,
        "l1_initial_pruned_accuracy": 0.83004,
        "l1_fine_tuned_accuracy": 0.85504,
        "rand_initial_pruned_accuracy": 0.54652,
        "rand_fine_tuned_accuracy": 0.80812
    },
    {
        "sparsity": 0.4,
        "l1_initial_pruned_accuracy": 0.80796,
        "l1_fine_tuned_accuracy": 0.8468,
        "rand_initial_pruned_accuracy": 0.50972,
        "rand_fine_tuned_accuracy": 0.78696
    },
    {
        "sparsity": 0.5,
        "l1_initial_pruned_accuracy": 0.76728,
        "l1_fine_tuned_accuracy": 0.84148,
        "rand_initial_pruned_accuracy": 0.5066,
        "rand_fine_tuned_accuracy": 0.74436
    },
    {
        "sparsity": 0.6,
        "l1_initial_pruned_accuracy": 0.563,
        "l1_fine_tuned_accuracy": 0.82284,
        "rand_initial_pruned_accuracy": 0.5046,
        "rand_fine_tuned_accuracy": 0.52732
    },
    {
        "sparsity": 0.7,
        "l1_initial_pruned_accuracy": 0.54052,
        "l1_fine_tuned_accuracy": 0.8098,
        "rand_initial_pruned_accuracy": 0.50292,
        "rand_fine_tuned_accuracy": 0.50436
    },
    {
        "sparsity": 0.8,
        "l1_initial_pruned_accuracy": 0.50076,
        "l1_fine_tuned_accuracy": 0.76204,
        "rand_initial_pruned_accuracy": 0.49896,
        "rand_fine_tuned_accuracy": 0.504
    },
    {
        "sparsity": 0.9,
        "l1_initial_pruned_accuracy": 0.50312,
        "l1_fine_tuned_accuracy": 0.56104,
        "rand_initial_pruned_accuracy": 0.50224,
        "rand_fine_tuned_accuracy": 0.5034
    }
]

import matplotlib.pyplot as plt

def plot_pruning_results_separated(results):
    # Extract sparsity levels
    sparsities = [res["sparsity"] for res in results]
    
    # Extract accuracies
    l1_initial = [res["l1_initial_pruned_accuracy"] for res in results]
    l1_fine_tuned = [res["l1_fine_tuned_accuracy"] for res in results]
    rand_initial = [res["rand_initial_pruned_accuracy"] for res in results]
    rand_fine_tuned = [res["rand_fine_tuned_accuracy"] for res in results]

    # Calculate highest achieved accuracy per sparsity
    highest_accuracies = [
        max(filter(None, [l1_init, l1_fine, rand_init, rand_fine]))
        for l1_init, l1_fine, rand_init, rand_fine in zip(l1_initial, l1_fine_tuned, rand_initial, rand_fine_tuned)
    ]

    # Plot 1: Highest achieved accuracy per sparsity
    plt.figure(figsize=(10, 6))
    plt.plot(
        sparsities,
        highest_accuracies,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Highest Accuracy",
    )
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.xlabel("Sparsity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Highest Achieved Accuracy by Sparsity", fontsize=14, pad=15)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    for x, y in zip(sparsities, highest_accuracies):
        plt.text(x, y + 0.005, f"{y*100:.1f}%", fontsize=8, ha="center")
    plt.tight_layout()
    plt.savefig("highest_accuracy_by_sparsity.png", dpi=300)
    plt.show()

    # Plot 2: Random and L1-Norm accuracy curves
    plt.figure(figsize=(10, 6))
    rand_accuracies = [
        max(filter(None, [rand_init, rand_fine]))
        for rand_init, rand_fine in zip(rand_initial, rand_fine_tuned)
    ]
    l1_accuracies = [
        max(filter(None, [l1_init, l1_fine]))
        for l1_init, l1_fine in zip(l1_initial, l1_fine_tuned)
    ]
    plt.plot(
        sparsities,
        rand_accuracies,
        marker="s",
        linestyle="-.",
        linewidth=2,
        label="Random Pruning",
    )
    plt.plot(
        sparsities,
        l1_accuracies,
        marker="d",
        linestyle=":",
        linewidth=2,
        label="L1-Norm Pruning",
    )
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.xlabel("Sparsity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Effect of Sparsity on Pruning Accuracy", fontsize=14, pad=15)
    plt.legend(fontsize=10, loc="lower left")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    for x, y in zip(sparsities, rand_accuracies):
        plt.text(x, y + 0.005, f"{y*100:.1f}%", fontsize=8, ha="center")
    for x, y in zip(sparsities, l1_accuracies):
        plt.text(x, y + 0.005, f"{y*100:.1f}%", fontsize=8, ha="center")
    plt.tight_layout()
    plt.savefig("pruning_accuracy_by_sparsity.png", dpi=300)
    plt.show()

# Call the function
plot_pruning_results_separated(results)

# {'sparsity': 0.2, 'l1_initial_pruned_accuracy': 0.83992, 'l1_fine_tuned_accuracy': 0.85832, 'rand_initial_pruned_accuracy': 0.6308, 'rand_fine_tuned_accuracy': 0.8232}
# {'sparsity': 0.3, 'l1_initial_pruned_accuracy': 0.83004, 'l1_fine_tuned_accuracy': 0.85504, 'rand_initial_pruned_accuracy': 0.54652, 'rand_fine_tuned_accuracy': 0.80812}
# {'sparsity': 0.4, 'l1_initial_pruned_accuracy': 0.80796, 'l1_fine_tuned_accuracy': 0.8468, 'rand_initial_pruned_accuracy': 0.50972, 'rand_fine_tuned_accuracy': 0.78696}
# {'sparsity': 0.5, 'l1_initial_pruned_accuracy': 0.76728, 'l1_fine_tuned_accuracy': 0.84148, 'rand_initial_pruned_accuracy': 0.5066, 'rand_fine_tuned_accuracy': 0.74436}
# {'sparsity': 0.6, 'l1_initial_pruned_accuracy': 0.563, 'l1_fine_tuned_accuracy': 0.82284, 'rand_initial_pruned_accuracy': 0.5046, 'rand_fine_tuned_accuracy': 0.52732}
# {'sparsity': 0.7, 'l1_initial_pruned_accuracy': 0.54052, 'l1_fine_tuned_accuracy': 0.8098, 'rand_initial_pruned_accuracy': 0.50292, 'rand_fine_tuned_accuracy': 0.50436}
# {'sparsity': 0.8, 'l1_initial_pruned_accuracy': 0.50076, 'l1_fine_tuned_accuracy': 0.76204, 'rand_initial_pruned_accuracy': 0.49896, 'rand_fine_tuned_accuracy': 0.504}
# {'sparsity': 0.9, 'l1_initial_pruned_accuracy': 0.50312, 'l1_fine_tuned_accuracy': 0.56104, 'rand_initial_pruned_accuracy': 0.50224, 'rand_fine_tuned_accuracy': 0.5034}