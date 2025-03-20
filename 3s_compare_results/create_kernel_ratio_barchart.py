import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MaxNLocator

# Directory containing the CSV files
gpu = 'A30'
data_dir = f'gt_time_csv_{gpu}'

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

# Use custom order for datasets
custom_order = ['reddit', 'yelp', 'citeseer', 'pubmed', 'cora', 'igb_small', 'ogbn-arxiv', 'flickr', 'Questions']
# Filter custom_order to only include datasets that are actually present
datasets = [dataset for dataset in custom_order if dataset in datasets]

# Define methods and hidden dimensions
methods = ['dgl', 'f3s', 'dfgnn_tiling', 'flashSparse']
hidden_dims = [64, 128, 256]

# Define display names for methods
method_display_names = {
    'dgl': 'DGL',
    'f3s': 'Fused3S',
    'dfgnn_tiling': 'DF-GNN',
    'flashSparse': 'FlashSparse'
}

# Define custom colors for methods
method_colors = {
    'dgl': '#808080',  # Grey for dgl
    'f3s': '#1f77b4',  # Blue for f3s
    'dfgnn_tiling': '#2ca02c',  # Green for dfgnn
    'flashSparse': '#d62728'  # Red for flashSparse
}

# Define markers for methods
method_markers = {
    'dgl': 'o',        # Circle for dgl
    'f3s': 's',        # Square for f3s
    'dfgnn_tiling': '^', # Triangle for dfgnn
    'flashSparse': 'D'  # Diamond for flashSparse
}

# Prepare data for plotting
plot_data = []

for dataset in datasets:
    # Load total time data and kernel time data
    total_file = f'gt_times_total_1heads_{dataset}.csv'
    kernel_file = f'gt_times_kernel_1heads_{dataset}.csv'
    
    if total_file in csv_files and kernel_file in csv_files:
        total_df = pd.read_csv(os.path.join(data_dir, total_file), index_col=0)
        kernel_df = pd.read_csv(os.path.join(data_dir, kernel_file), index_col=0)
        
        # For each hidden dimension
        for dim in hidden_dims:
            dim_str = str(dim)
            
            # Calculate ratio for each method
            for method in methods:
                if method in total_df.index and method in kernel_df.index:
                    if dim_str in total_df.columns and dim_str in kernel_df.columns:
                        total_time = total_df.loc[method, dim_str]
                        kernel_time = kernel_df.loc[method, dim_str]
                        
                        # Skip if total_time is 0 (indicating no data)
                        if total_time == 0:
                            continue
                        
                        # Calculate kernel to total time ratio
                        kernel_ratio = kernel_time / total_time
                        
                        # Add to plot data
                        plot_data.append({
                            'dataset': dataset,
                            'method': method,
                            'method_display': method_display_names[method],
                            'hidden_dim': dim,
                            'kernel_ratio': kernel_ratio,
                            'total_time': total_time,
                            'kernel_time': kernel_time
                        })

# Convert to DataFrame for easier plotting
plot_df = pd.DataFrame(plot_data)

# Increase font sizes
plt.rcParams.update({'font.size': 20})  # Increased from 18 to 20

# Create the plot
plt.figure(figsize=(30, 5.5))

# Create a mapping for better dataset names display
dataset_display_names = {dataset: dataset.replace('_', ' ').title() for dataset in datasets}

# Calculate x positions for each dataset and dimension
x_positions_all = []
x_ticks = []
x_tick_labels = []

current_pos = 1
for dataset_idx, dataset in enumerate(datasets):
    dataset_start_pos = current_pos
    
    for dim_idx, dim in enumerate(hidden_dims):
        x_ticks.append(current_pos)
        x_tick_labels.append(str(dim))
        
        # If this is the first dimension of a dataset, add dataset label
        if dim_idx == 0:
            plt.text(current_pos + len(hidden_dims)/2 - 0.5, -0.1, dataset_display_names[dataset], 
                    ha='center', va='top', fontsize=22, fontweight='bold', rotation=0)
        
        current_pos += 1
    
    # Add space between datasets
    current_pos += 1

# Plot lines for each method within each dataset
for method in methods:
    # For legend
    plt.plot([], [], color=method_colors[method], marker=method_markers[method], 
             linestyle='-', linewidth=2, markersize=8, 
             label=method_display_names[method])
    
    # Plot data for each dataset
    current_pos = 1
    for dataset in datasets:
        dataset_method_data = plot_df[(plot_df['dataset'] == dataset) & (plot_df['method'] == method)]
        
        if not dataset_method_data.empty:
            # Sort by hidden dimension
            dataset_method_data = dataset_method_data.sort_values('hidden_dim')
            
            # Get x positions for this dataset's hidden dimensions
            x_positions = []
            y_values = []
            
            for dim_idx, dim in enumerate(hidden_dims):
                dim_data = dataset_method_data[dataset_method_data['hidden_dim'] == dim]
                if not dim_data.empty:
                    x_pos = current_pos
                    x_positions.append(x_pos)
                    y_values.append(dim_data['kernel_ratio'].values[0])
                current_pos += 1
            
            # Skip the spacing between datasets in position calculation
            current_pos += 1
            
            # If only one data point (e.g., for DGL), just plot a point
            if len(x_positions) == 1:
                plt.plot(x_positions[0], y_values[0], color=method_colors[method], 
                         marker=method_markers[method], markersize=10)
            # Otherwise plot a line connecting the points for this method and dataset
            elif len(x_positions) > 1:
                plt.plot(x_positions, y_values, color=method_colors[method], 
                         marker=method_markers[method], linestyle='-', linewidth=2, 
                         markersize=8)
        else:
            # Skip this dataset's positions if no data
            current_pos += len(hidden_dims) + 1

# Set x-axis ticks and labels
plt.xticks(x_ticks, x_tick_labels, fontsize=20)

# Add legend and labels
if gpu == 'A30':
  plt.legend(loc='upper right', ncol=len(methods), fontsize=20)
# else:
#   plt.legend(loc='lower right', ncol=len(methods), fontsize=20)
plt.ylabel('Kernel Time / Total Time Ratio', fontsize=22)

# Create more y-ticks (10 ticks from 0 to 1)
y_ticks = np.linspace(0, 1, 11)
plt.yticks(y_ticks, fontsize=20)

# Add horizontal grid lines for each y-tick
plt.grid(axis='y', linestyle='-', alpha=0.9, color='#CCCCCC', linewidth=0.8)

# Add vertical grid lines to separate datasets
current_pos = 1
for dataset in datasets:
    # Add a vertical line after each dataset's dimensions
    plt.axvline(x=current_pos + len(hidden_dims), color='#AAAAAA', linestyle='-', alpha=0.3)
    current_pos += len(hidden_dims) + 1

# Ensure y-axis starts at 0 and ends at 1 (ratio is between 0 and 1)
plt.ylim(bottom=0, top=1.05)

# Set x-axis limits
plt.xlim(0.5, current_pos - 1.5)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.01, right=0.99, top=0.95)  # Increased bottom margin for dataset labels
plt.savefig(data_dir + '/kernel_ratio_lineplot_' + gpu + '.png', dpi=300, bbox_inches='tight')
plt.close()

print("Chart created and saved as 'kernel_ratio_lineplot_" + gpu + ".png'") 