import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

def plot_combined_optuna_results(csv_paths):
    """
    Loads multiple Optuna study results from CSV files, computes the running max accuracy 
    for each, and creates a combined plot for all search methods.
    """
    plt.figure(figsize=(10, 6))

    for csv_path in csv_paths:
        # Load the CSV file
        data = pd.read_csv(csv_path)
        
        # Extract accuracy values
        accuracies = data["accuracy"].tolist()
        
        # Compute the running maximum
        running_max = []
        current_max = float('-inf')
        for acc in accuracies:
            current_max = max(current_max, acc)
            running_max.append(current_max)
        
        # Determine the label based on the file name
        if 'random' in csv_path.lower():
            label = 'Random Search'
        elif 'grid' in csv_path.lower():
            label = 'Grid Search'
        elif 'tpe' in csv_path.lower():
            label = 'TPE Search'
        else:
            label = 'Unknown Search Method'
        
        # Plot the running maximum for the current search method
        plt.plot(
            range(len(running_max)),
            running_max,
            marker="o",
            linestyle="--",
            linewidth=2,
            label=label,
        )
    
    # Grid and labels
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.xlabel("Number of Trials", fontsize=12)
    plt.ylabel("Maximum Achieved Accuracy", fontsize=12)
    plt.title("Comparison of Maximum Achieved Accuracy by Search Method", fontsize=14, pad=15)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    
    # Legend and layout
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig('combined_optuna_results.png', dpi=300)
    plt.show()


# Example usage: Replace these file paths with your actual CSV file paths
csv_paths = [
    'random_study_trials.csv',
    'grid_study_trials.csv',
    'tpe_study_trials.csv',
]

plot_combined_optuna_results(csv_paths)
