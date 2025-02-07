import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# # Function to load results from CSV
# def load_results_from_csv(filepath):
#     return pd.read_csv(filepath)

# # Load the results
# csv_filepath = "formatted_training_results.csv"
# results_df = load_results_from_csv(csv_filepath)


# def plot_compression_aware_results(results):
#     import matplotlib.ticker as mticker

#     # Separate results by complexity type
#     no_compression = [res for res in results if res["complexity"] == "No Compression"]
#     compression_no_post = [res for res in results if res["complexity"] == "Compression-aware (No Post-Training)"]
#     compression_with_post = [res for res in results if res["complexity"] == "Compression-aware (Post-Training)"]

#     # Helper to compute running max
#     def get_running_max(data):
#         running_max = []
#         current_max = float('-inf')
#         for res in data:
#             current_max = max(current_max, res["value"])
#             running_max.append(current_max)
#         return running_max

#     # Compute running max for each scenario
#     no_compression_max = get_running_max(no_compression)
#     compression_no_post_max = get_running_max(compression_no_post)
#     compression_with_post_max = get_running_max(compression_with_post)

#     # Trials count
#     trials_no_compression = list(range(len(no_compression_max)))
#     trials_compression_no_post = list(range(len(compression_no_post_max)))
#     trials_compression_with_post = list(range(len(compression_with_post_max)))

#     # Plot the results
#     plt.figure(figsize=(10, 6))
    
#     # No Compression
#     plt.plot(
#         trials_no_compression,
#         no_compression_max,
#         marker="o",
#         linestyle="--",
#         linewidth=2,
#         label="No Compression",
#     )
    
#     # Compression-aware (No Post-Training)
#     plt.plot(
#         trials_compression_no_post,
#         compression_no_post_max,
#         marker="s",
#         linestyle="-.",
#         linewidth=2,
#         label="Compression-aware (No Post-Training)",
#     )
    
#     # Compression-aware (Post-Training)
#     plt.plot(
#         trials_compression_with_post,
#         compression_with_post_max,
#         marker="d",
#         linestyle=":",
#         linewidth=2,
#         label="Compression-aware (Post-Training)",
#     )

#     # Format the plot
#     plt.grid(visible=True, linestyle="--", alpha=0.5)
#     plt.xlabel("Number of Trials", fontsize=12)
#     plt.ylabel("Maximum Achieved Accuracy", fontsize=12)
#     plt.title("Compression-Aware Search Results", fontsize=14, pad=15)
    
#     # Format y-axis as percentage
#     plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    
#     # Add annotations for each point
#     for i, val in enumerate(no_compression_max):
#         plt.text(i, val, f"{val*100:.1f}%", fontsize=8, ha="center", va="bottom")
#     for i, val in enumerate(compression_no_post_max):
#         plt.text(i, val, f"{val*100:.1f}%", fontsize=8, ha="center", va="bottom")
   


# plot_compression_aware_results(results_p2)

# Function to load results from CSV
def load_results_from_csv(filepath):
    return pd.read_csv(filepath)
 
# Load the results
csv_filepath = "formatted_training_results.csv"
results_df = load_results_from_csv(csv_filepath)

# Function to plot the compression-aware results
def plot_compression_aware_results(results_df):
    # Separate results by complexity type
    no_compression = results_df[results_df["complexity"] == "No Compression"]
    compression_no_post = results_df[results_df["complexity"] == "Compression-aware (No Post-Training)"]
    compression_with_post = results_df[results_df["complexity"] == "Compression-aware (Post-Training)"]

    # Helper to compute running max
    def get_running_max(values):
        running_max = []
        current_max = float('-inf')
        for val in values:
            current_max = max(current_max, val)
            running_max.append(current_max)
        return running_max

    # Compute running max for each scenario
    no_compression_max = get_running_max(no_compression["value"].tolist())
    compression_no_post_max = get_running_max(compression_no_post["value"].tolist())
    compression_with_post_max = get_running_max(compression_with_post["value"].tolist())

    # Trials count
    trials_no_compression = no_compression["trial"].tolist()
    trials_compression_no_post = compression_no_post["trial"].tolist()
    trials_compression_with_post = compression_with_post["trial"].tolist()

    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # No Compression
    plt.plot(
        trials_no_compression,
        no_compression_max,
        marker="o",
        linestyle="--",
        linewidth=2,
        label="No Compression",
    )
    
    # Compression-aware (No Post-Training)
    plt.plot(
        trials_compression_no_post,
        compression_no_post_max,
        marker="s",
        linestyle="-.",
        linewidth=2,
        label="Compression-aware (No Post-Training)",
    )
    
    # Compression-aware (Post-Training)
    plt.plot(
        trials_compression_with_post,
        compression_with_post_max,
        marker="d",
        linestyle=":",
        linewidth=2,
        label="Compression-aware (Post-Training)",
    )

    # Format the plot
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Maximum Achieved Accuracy", fontsize=12)
    plt.title("Compression-Aware Search Results", fontsize=14, pad=15)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    
    # Add annotations for each point
    for i, val in zip(trials_no_compression, no_compression_max):
        plt.text(i, val, f"{val*100:.1f}%", fontsize=8, ha="center", va="bottom")
    for i, val in zip(trials_compression_no_post, compression_no_post_max):
        plt.text(i, val, f"{val*100:.1f}%", fontsize=8, ha="center", va="bottom")
    for i, val in zip(trials_compression_with_post, compression_with_post_max):
        plt.text(i, val, f"{val*100:.1f}%", fontsize=8, ha="center", va="bottom")

    # Show legend
    plt.legend()
    plt.tight_layout()
    plt.savefig("compression_aware_results.png", dpi=300)
    plt.show()

# Plot the results using the updated function
plot_compression_aware_results(results_df)
