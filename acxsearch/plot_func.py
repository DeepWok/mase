
import numpy as np
import matplotlib.pyplot as plt


# Call the function with the loaded acc_list




# Example usage:
# file_paths = [
#     "/home/cx922/mase/deit_tiny_gelu_search_results_int_1.pkl",
#     "/home/cx922/mase/deit_tiny_gelu_search_results_int_2.pkl",
#     "/home/cx922/mase/deit_tiny_gelu_search_results_int_3.pkl"
# ]
# plot_gelu_accuracy_loss(file_paths)


import torch
import matplotlib.pyplot as plt
import math

# Create data points (avoiding 0 to prevent division by zero)
x = torch.linspace(-5, 5, 1000)

# Calculate GELU function using torch.nn.functional.gelu
y_gelu = torch.nn.functional.gelu(x)
# Calculate ReLU function
y_relu = torch.nn.functional.relu(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y_gelu.numpy(), 'b-', label='GELU(x)')
plt.plot(x.numpy(), y_relu.numpy(), 'r-', label='ReLU(x)')
plt.grid(False)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Add vertical dashed lines at x=-2 and x=2
plt.axvline(x=-2, color='r', linestyle='--', linewidth=2, alpha=0.7)
plt.axvline(x=2, color='r', linestyle='--', linewidth=2, alpha=0.7)

# Add shaded regions outside of (-2, 2)
plt.axvspan(-5, -2, alpha=0.2, color='gray')
plt.axvspan(2, 5, alpha=0.2, color='gray')

# Plot hash points for hash_bit = 3
hash_bit_3 = 3
hash_in_3 = torch.linspace(1/2**hash_bit_3, 0.5, 2**(hash_bit_3 - 1) - 1)
for point in hash_in_3:
    # Calculate the y value using torch.nn.functional.gelu
    y_val = torch.nn.functional.gelu(point)
    # Plot the point on the curve
    plt.plot(point.item(), y_val.item(), 'ro', markersize=6)
    # Draw a line from x-axis to the curve
    plt.plot([point.item(), point.item()], [0, y_val.item()], 'r--', alpha=0.7)

# Plot hash points for hajksh_bit = 4
hash_bit_4 = 4
hash_in_4 = torch.linspace(0.5 + 1/2**hash_bit_4, 1, 2**(hash_bit_4 - 1) - 1)
for point in hash_in_4:
    # Calculate the y value using torch.nn.functional.gelu
    y_val = torch.nn.functional.gelu(point)
    # Plot the point on the curve
    plt.plot(point.item(), y_val.item(), 'go', markersize=6)
    # Draw a line from x-axis to the curve
    plt.plot([point.item(), point.item()], [0, y_val.item()], 'g--', alpha=0.7)

# Add labels and title
plt.xlabel('x')
plt.ylabel('Value')
plt.title('GELU and ReLU Functions with Hash Input Points')
plt.legend(['GELU(x)', 'ReLU(x)', 'Range Boundary'])

# Set reasonable y-axis limits to better show the function behavior
plt.ylim(-1, 5)

# Show the plot
plt.show()
