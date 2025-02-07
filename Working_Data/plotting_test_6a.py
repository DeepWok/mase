import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

def plot_optuna_search_results(csv_filepath):
    # Read the CSV file containing the trial results
    results_df = pd.read_csv(csv_filepath)
    
    # Ensure the trials are in order (by trial_number)
    results_df = results_df.sort_values(by='trial_number')
    
    # Compute the running maximum accuracy up to each trial
    running_max = []
    current_max = float('-inf')
    for acc in results_df['accuracy']:
        current_max = max(current_max, acc)
        running_max.append(current_max)
    results_df['running_max'] = running_max
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['trial_number'], results_df['running_max'],
             marker='o', linestyle='-', linewidth=2)
    
    # Label the axes and the plot
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Maximum Achieved Accuracy", fontsize=12)
    plt.title("Optuna Search Results", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Format the y-axis to display percentages
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    
    # Annotate each point with its percentage value
    for trial, acc in zip(results_df['trial_number'], results_df['running_max']):
        plt.text(trial, acc, f"{acc*100:.1f}%", fontsize=8, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("optuna_search_results.png", dpi=300)
    plt.show()

plot_optuna_search_results("optuna_results.csv")

