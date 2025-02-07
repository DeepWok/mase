import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

def plot_extended_optuna_search_results(csv_filepath):
    """
    Plots both the running max accuracy and the accuracy reached at each iteration.
    """
    # Read the CSV file containing the trial results
    results_df = pd.read_csv(csv_filepath)
    
    # Ensure the trials are sorted by trial_number
    results_df = results_df.sort_values(by="trial_number")
    
    # Compute the running maximum accuracy up to each trial
    running_max = []
    current_max = float("-inf")
    for acc in results_df["accuracy"]:
        current_max = max(current_max, acc)
        running_max.append(current_max)
    results_df["running_max"] = running_max

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy reached at each iteration
    plt.plot(results_df["trial_number"], results_df["accuracy"], 
             marker="o", linestyle="-", linewidth=2, label="Accuracy per Trial", color="tab:blue")
    
    # Plot running max accuracy
    plt.plot(results_df["trial_number"], results_df["running_max"], 
             marker="s", linestyle="--", linewidth=2, label="Maximum Achieved Accuracy", color="tab:red")
    
    # Label the axes and the plot
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Mixed Precision Search", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Format the y-axis to display percentages
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))

    # Annotate the running max points with their percentage values
    for trial, acc in zip(results_df["trial_number"], results_df["running_max"]):
        plt.text(trial, acc, f"{acc*100:.1f}%", fontsize=8, ha="center", va="bottom")

    # Add legend
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("extended_optuna_search_results_2_curves.png", dpi=300)
    plt.show()

# def plot_extended_optuna_search_results(csv_filepath):
#     """
#     Plots the running max accuracy for the first method,
#     where multiple layer types are tested in a single search.
#     """
#     # Read the CSV file containing the trial results
#     results_df = pd.read_csv(csv_filepath)
    
#     # Ensure the trials are sorted by trial_number
#     results_df = results_df.sort_values(by="trial_number")
    
#     # Compute the running maximum accuracy up to each trial
#     running_max = []
#     current_max = float("-inf")
#     for acc in results_df["accuracy"]:
#         current_max = max(current_max, acc)
#         running_max.append(current_max)
#     results_df["running_max"] = running_max
    
#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(results_df["trial_number"], results_df["running_max"],
#              marker="o", linestyle="-", linewidth=2)
    
#     # Label the axes and the plot
#     plt.xlabel("Number of Trials", fontsize=12)
#     plt.ylabel("Maximum Achieved Accuracy", fontsize=12)
#     plt.title("Optuna Search Results - Mixed Layer Search", fontsize=14, pad=15)
#     plt.grid(True, linestyle="--", alpha=0.5)
    
#     # Format the y-axis to display percentages
#     plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    
#     # Annotate each point with its percentage value
#     for trial, acc in zip(results_df["trial_number"], results_df["running_max"]):
#         plt.text(trial, acc, f"{acc*100:.1f}%", fontsize=8, ha="center", va="bottom")
    
#     plt.tight_layout()
#     plt.savefig("extended_optuna_search_results.png", dpi=300)
#     plt.show()

def plot_extended_optuna_search_results_v2(csv_filepath):
    """
    Plots the running max accuracy for the second method,
    where each precision type is tested separately.
    Creates multiple curves, one per precision type.
    """
    # Read the CSV file
    results_df = pd.read_csv(csv_filepath)
    
    # Ensure trials are sorted correctly
    results_df = results_df.sort_values(by="trial_number")

    # Prepare figure
    plt.figure(figsize=(10, 6))

    # Process each precision type separately
    for precision, group in results_df.groupby("precision_type"):
        group = group.sort_values("trial_number")
        trial_nums = group["trial_number"].tolist()
        accuracies = group["trial_accuracy"].tolist()

        # Compute running maximum accuracy
        running_max = []
        current_max = float("-inf")
        for acc in accuracies:
            current_max = max(current_max, acc)
            running_max.append(current_max)

        # Plot the running maximum for this precision type
        plt.plot(
            trial_nums,
            running_max,
            marker="o",
            linestyle="--",
            linewidth=2,
            label=precision,  # Use precision type as the label
        )

    # Grid and labels
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Maximum Achieved Accuracy", fontsize=12)
    plt.title("Optuna Search Results - Per Precision Type", fontsize=14, pad=15)

    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    # plt.ylim(0.85, 0.9)
    # Legend and layout
    plt.legend(fontsize=10, loc="lower right", title="Precision Type")
    plt.tight_layout()
    plt.savefig("optuna_combined_precision_progress.png", dpi=300)
    plt.show()


# Run both plots
plot_extended_optuna_search_results("extended_optuna_results.csv")
plot_extended_optuna_search_results_v2("optuna_combined_results.csv")
