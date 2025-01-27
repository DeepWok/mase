import matplotlib.pyplot as plt

# (Full-Precision) First test is full precision model, 
# (PTQ) Second test is quantised model accuracy evaluation (no retraining), 
# (QAT) Third test is quantised model accuracy evaluation (with retraining)

results = [
    {
        "test_id": 1,
        "total_fixed_points": 12,
        "data_in_width": 8,
        "data_in_frac_width": 4,
        "weight_width": 8,
        "weight_frac_width": 4,
        "bias_width": 8,
        "bias_frac_width": 4,
        "full_precision_accuracy": 0.83544,
        "ptq_accuracy": 0.78388,
        "qat_accuracy": 0.84076,
    },
    {
        "test_id": 2,
        "total_fixed_points": 4,
        "data_in_width": 2,
        "data_in_frac_width": 2,
        "weight_width": 2,
        "weight_frac_width": 2,
        "bias_width": 2,
        "bias_frac_width": 2,
        "full_precision_accuracy": 0.83544,
        "ptq_accuracy": None,
        "qat_accuracy": None,
    },
    {
        "test_id": 3,
        "total_fixed_points": 20,
        "data_in_width": 12,
        "data_in_frac_width": 8,
        "weight_width": 12,
        "weight_frac_width": 8,
        "bias_width": 12,
        "bias_frac_width": 8,
        "full_precision_accuracy": 0.83544,
        "ptq_accuracy": None,
        "qat_accuracy": None,
    },
    {
        "test_id": 4,
        "total_fixed_points": 32,
        "data_in_width": 16,
        "data_in_frac_width": 16,
        "weight_width": 16,
        "weight_frac_width": 16,
        "bias_width": 16,
        "bias_frac_width": 16,
        "full_precision_accuracy": 0.83544,
        "ptq_accuracy": None,
        "qat_accuracy": None,
    },
]

def plot_results(results):
    fixed_point_widths = [res["total_fixed_points"] for res in results]
    ptq_accuracies = [res["ptq_accuracy"] for res in results if res["ptq_accuracy"] is not None]
    qat_accuracies = [res["qat_accuracy"] for res in results if res["qat_accuracy"] is not None]

    # Total Width vs Accuracy curve
    highest_accuracies = [max(ptq_accuracies[i], qat_accuracies[i]) for i in range(len(ptq_accuracies))]
    plt.plot(
        fixed_point_widths[:len(highest_accuracies)],
        highest_accuracies,
        marker="o",
        label="Total Width vs Highest Accuracy",
    )

    # PTQ curve
    plt.plot(
        fixed_point_widths[:len(ptq_accuracies)],
        ptq_accuracies,
        marker="o",
        label="PTQ Accuracy",
    )
    
    # QAT curve
    plt.plot(
        fixed_point_widths[:len(qat_accuracies)],
        qat_accuracies,
        marker="o",
        label="QAT Accuracy",
    )
    
    # Add labels and legend
    plt.xlabel("Fixed Point Width")
    plt.ylabel("Accuracy")
    plt.title("Effect of Fixed Point Width on Accuracy")
    plt.legend()
    plt.grid()
    plt.show()