import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# (Full-Precision) First test is full precision model, 
# (PTQ) Second test is quantised model accuracy evaluation (no retraining), 
# (QAT) Third test is quantised model accuracy evaluation (with retraining)

results = [
    {
        "test_id": 1,
        "data_in_width": 4,
        "data_in_frac_width": 2,
        "weight_width": 4,
        "weight_frac_width": 2,
        "bias_width": 4,
        "bias_frac_width": 2,
        "full_precision_accuracy": 0.83732,
        "ptq_accuracy": 0.5,
        "qat_accuracy": 0.5,
    },
    {
        "test_id": 2,
        "data_in_width": 8,
        "data_in_frac_width": 4,
        "weight_width": 8,
        "weight_frac_width": 4,
        "bias_width": 8,
        "bias_frac_width": 4,
        "full_precision_accuracy": 0.83732,
        "ptq_accuracy": 0.78388,
        "qat_accuracy": 0.84076,
    },
    {
        "test_id": 5,
        "data_in_width": 12,
        "data_in_frac_width": 8,
        "weight_width": 12,
        "weight_frac_width": 8,
        "bias_width": 12,
        "bias_frac_width": 8,
        "full_precision_accuracy": 0.83732,
        "ptq_accuracy": 0.83704,
        "qat_accuracy": 0.84412,
    },
    {
        "test_id": 6,
        "data_in_width": 16,
        "data_in_frac_width": 8,
        "weight_width": 16,
        "weight_frac_width": 8,
        "bias_width": 16,
        "bias_frac_width": 8,
        "full_precision_accuracy": 0.83732,
        "ptq_accuracy": 0.83796,
        "qat_accuracy": 0.84468,
    },
    {
        "test_id": 6,
        "data_in_width": 20,
        "data_in_frac_width": 10,
        "weight_width": 20,
        "weight_frac_width": 10,
        "bias_width": 20,
        "bias_frac_width": 10,
        "full_precision_accuracy": 0.83732,
        "ptq_accuracy": 0.83776,
        "qat_accuracy": 0.845,
    },
    {
        "test_id": 6,
        "data_in_width": 24,
        "data_in_frac_width": 12,
        "weight_width": 24,
        "weight_frac_width": 12,
        "bias_width": 24,
        "bias_frac_width": 12,
        "full_precision_accuracy": 0.83732,
        "ptq_accuracy": 0.83732,
        "qat_accuracy": 0.84488,
    },
    {
        "test_id": 7,
        "data_in_width": 28,
        "data_in_frac_width": 14,
        "weight_width": 28,
        "weight_frac_width": 14,
        "bias_width": 28,
        "bias_frac_width": 14,
        "full_precision_accuracy": 0.83732,
        "ptq_accuracy": 0.83736,
        "qat_accuracy": 0.84504,
    },
    {
        "test_id": 7,
        "data_in_width": 32,
        "data_in_frac_width": 16,
        "weight_width": 32,
        "weight_frac_width": 16,
        "bias_width": 32,
        "bias_frac_width": 16,
        "full_precision_accuracy": 0.83732,
        "ptq_accuracy": 0.83736,
        "qat_accuracy": 0.84492,
    }]

def plot_results_separated(results):
    fixed_point_widths = [4, 8, 12, 16, 20, 24, 28, 32]
    ptq_accuracies = [res["ptq_accuracy"] for res in results if res["ptq_accuracy"] is not None]
    qat_accuracies = [res["qat_accuracy"] for res in results if res["qat_accuracy"] is not None]
    highest_accuracies = [max(ptq_accuracies[i], qat_accuracies[i]) for i in range(len(ptq_accuracies))]

    # Plot 1: PTQ vs QAT
    plt.figure(figsize=(10, 6))
    plt.plot(
        fixed_point_widths[:len(ptq_accuracies)],
        ptq_accuracies,
        marker="s",
        label="PTQ Accuracy",
        linestyle="-.",
        linewidth=2,
        alpha=0.8,
    )
    plt.plot(
        fixed_point_widths[:len(qat_accuracies)],
        qat_accuracies,
        marker="d",
        label="QAT Accuracy",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
    )
    # Add labels, grid, and legend
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.xlabel("Fixed Point Width", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("PTQ vs QAT Accuracy by Fixed Point Width", fontsize=14, pad=15)
    plt.legend(fontsize=10, loc="lower right")

    # Format y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    # Add value labels on the points
    for x, y in zip(fixed_point_widths[:len(ptq_accuracies)], ptq_accuracies):
        plt.text(x, y + 0.002, f"{y*100:.2f}%", fontsize=8, ha="center")
    for x, y in zip(fixed_point_widths[:len(qat_accuracies)], qat_accuracies):
        plt.text(x, y + 0.002, f"{y*100:.2f}%", fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig("ptq_vs_qat_accuracy.png", dpi=300)
    plt.show()

    # Plot 2: Highest Accuracy vs Fixed Width
    plt.figure(figsize=(10, 6))
    plt.plot(
        fixed_point_widths[:len(highest_accuracies)],
        highest_accuracies,
        marker="o",
        label="Highest Accuracy",
        linestyle="--",
        linewidth=2,
    )

    # Add labels, grid, and legend
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.xlabel("Fixed Point Width", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Highest Accuracy by Fixed Point Width", fontsize=14, pad=15)

    # Format y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    # Add value labels on the points
    for x, y in zip(fixed_point_widths[:len(highest_accuracies)], highest_accuracies):
        plt.text(x, y + 0.002, f"{y*100:.2f}%", fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig("fixed_point_width_vs_accuracy.png", dpi=300)
    plt.show()

plot_results_separated(results)