import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MaxNLocator

# Directory containing the CSV files
data_dir = '.'

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Extract unique datasets
dataset_pattern = re.compile(r'gt_times_total_1heads_(.+)\.csv')
datasets = set()
for file in csv_files:
    match = dataset_pattern.match(file)
    if match:
        datasets.add(match.group(1))

# Remove ogbn-products from datasets
if 'ogbn-products' in datasets:
    datasets.remove('ogbn-products')

# Sort datasets for consistent ordering
datasets = sorted(list(datasets))

# Define methods and hidden dimensions
methods = ['dgl', 'f3s', 'dfgnn_tiling', 'flashSparse']
baseline_method = 'dfgnn_tiling'  # Method to use as baseline for speedup calculation
hidden_dims = [64, 128, 256]

# Define display names for methods
method_display_names = {
    'dgl': 'dgl',
    'f3s': 'f3s',
    'dfgnn_tiling': 'dfgnn',
    'flashSparse': 'flashSparse'
}

# Define custom colors for methods
method_colors = {
    'dgl': '#808080',  # Grey for dgl
    'f3s': '#1f77b4',  # Blue for f3s
    'dfgnn_tiling': '#2ca02c',  # Green for dfgnn
    'flashSparse': '#d62728'  # Red for flashSparse
}

# Prepare data for plotting
plot_data = []

for dataset in datasets:
    # Load kernel time data
    kernel_file = f'gt_times_kernel_1heads_{dataset}.csv'
    
    if kernel_file in csv_files:
        kernel_df = pd.read_csv(os.path.join(data_dir, kernel_file), index_col=0)
        
        # For each hidden dimension, calculate speedup relative to baseline
        for dim in hidden_dims:
            dim_str = str(dim)
            if dim_str in kernel_df.columns and baseline_method in kernel_df.index:
                baseline_kernel_time = kernel_df.loc[baseline_method, dim_str]
                
                # Skip if baseline time is 0 or not available
                if baseline_kernel_time == 0:
                    continue
                
                # Calculate speedup for each method
                for method in methods:
                    if method in kernel_df.index:
                        kernel_time = kernel_df.loc[method, dim_str]
                        
                        # Skip if kernel time is 0 (indicating no data)
                        if kernel_time == 0:
                            continue
                        
                        # Calculate speedup (baseline / method)
                        # For the baseline itself, this will be 1.0
                        speedup = baseline_kernel_time / kernel_time
                        
                        # Add to plot data
                        plot_data.append({
                            'dataset': dataset,
                            'method': method,
                            'method_display': method_display_names[method],
                            'hidden_dim': dim,
                            'speedup': speedup,
                            'kernel_time': kernel_time
                        })

# Convert to DataFrame for easier plotting
plot_df = pd.DataFrame(plot_data)

# Increase font sizes
plt.rcParams.update({'font.size': 14})  # Increase base font size

# Create the plot
plt.figure(figsize=(20, 10))

# Calculate the number of groups and width of bars
n_datasets = len(datasets)
n_dims = len(hidden_dims)
n_methods = len(methods)
group_width = 0.9  # Increased from 0.8 to reduce whitespace
bar_width = group_width / n_methods

# Generate x positions for bars
x_positions = []
x_ticks = []
x_tick_labels = []

# Create a mapping for better dataset names display
dataset_display_names = {dataset: dataset.replace('_', ' ').title() for dataset in datasets}

current_pos = 0.5  # Start closer to the left edge (reduced from default)
for d_idx, dataset in enumerate(datasets):
    dataset_label_pos = []  # To store positions for dataset labels
    
    for h_idx, dim in enumerate(hidden_dims):
        # Filter data for this dataset and dimension
        dataset_dim_data = plot_df[(plot_df['dataset'] == dataset) & (plot_df['hidden_dim'] == dim)]
        
        if not dataset_dim_data.empty:
            group_start_pos = current_pos
            
            # Add positions for each method in this group
            for m_idx, method in enumerate(methods):
                method_data = dataset_dim_data[dataset_dim_data['method'] == method]
                if not method_data.empty:
                    pos = current_pos + m_idx * bar_width
                    x_positions.append(pos)
                    
                    # Plot the speedup bar
                    speedup = method_data['speedup'].values[0]
                    
                    # Use custom colors for methods
                    color = method_colors[method]
                    
                    # Get display name for the method
                    display_name = method_data['method_display'].values[0]
                    
                    # For the baseline method, always show exactly 1.0
                    if method == baseline_method:
                        plt.bar(pos, 1.0, bar_width, color=color, 
                               label=display_name if d_idx == 0 and h_idx == 0 else "")
                        # Add a horizontal line at y=1 for reference
                        if d_idx == 0 and h_idx == 0 and m_idx == 0:
                            plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
                    else:
                        plt.bar(pos, speedup, bar_width, color=color,
                               label=display_name if d_idx == 0 and h_idx == 0 else "")
                    
                    # Add the actual kernel time value as text on the bar
                    kernel_time = method_data['kernel_time'].values[0]
                    plt.text(pos, speedup + 0.05, f"{kernel_time:.1f}", 
                            ha='center', va='bottom', fontsize=10, rotation=90)
            
            # Add tick for this dimension
            x_ticks.append(current_pos + (n_methods * bar_width) / 2 - bar_width/2)
            x_tick_labels.append(f"{dim}")
            
            dataset_label_pos.append(current_pos + (n_methods * bar_width) / 2 - bar_width/2)
            
            # Move to next group
            current_pos += n_methods * bar_width + bar_width * 0.5  # Reduced spacing between dimension groups
        
    # Add dataset label in the middle of its groups
    if dataset_label_pos:
        mid_pos = sum(dataset_label_pos) / len(dataset_label_pos)
        plt.text(mid_pos, -0.15, dataset_display_names[dataset], ha='center', va='top', 
                fontsize=16, fontweight='bold')  # Increased font size and moved closer to x-axis
    
    # Add space between datasets (reduced)
    current_pos += bar_width * 1.0  # Reduced from 2.0

# Set x-axis ticks and labels
plt.xticks(x_ticks, x_tick_labels, fontsize=14)

# Add legend and labels (no title)
plt.legend(loc='upper left', ncol=len(methods), fontsize=14)
plt.ylabel('Kernel Time Speedup (higher is better)', fontsize=16)
plt.grid(axis='y', linestyle='-', alpha=0.3)

# Ensure y-axis starts at 0
plt.ylim(bottom=0)
plt.yticks(fontsize=14)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, left=0.05, right=0.98, top=0.95)  # Adjusted margins
plt.savefig('kernel_speedup_barchart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Chart created and saved as 'kernel_speedup_barchart.png'") 