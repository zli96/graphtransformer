import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

# Directory containing the CSV files
gpu = 'H100'
if gpu == 'A30':
  data_dir = '.'
elif gpu == 'H100':
  data_dir = '../gt_time_csv_H100'

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Extract unique datasets
dataset_pattern = re.compile(r'gt_times_total_1heads_(.+)\.csv')
datasets = set()
for file in csv_files:
    match = dataset_pattern.match(file)
    if match:
        datasets.add(match.group(1))

# Remove ogbn-products and cluster from datasets
datasets_to_remove = ['ogbn-products', 'cluster']
for dataset in datasets_to_remove:
    if dataset in datasets:
        datasets.remove(dataset)

# Define custom order for datasets
custom_order = ['reddit', 'yelp', 'citeseer', 'pubmed', 'cora', 'igb_small', 'ogbn-arxiv', 'flickr', 'Questions']
# Filter and order datasets according to custom order
datasets = [dataset for dataset in custom_order if dataset in datasets]

# Define methods and hidden dimensions
methods = ['dgl', 'f3s', 'dfgnn_tiling', 'flashSparse']
baseline_method = 'dfgnn_tiling'  # Method to use as baseline for speedup calculation
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

# Prepare data for plotting
plot_data = []

for dataset in datasets:
    # Load total time data
    total_file = f'gt_times_total_1heads_{dataset}.csv'
    
    if total_file in csv_files:
        total_df = pd.read_csv(os.path.join(data_dir, total_file), index_col=0)
        
        # For each hidden dimension, calculate speedup relative to baseline
        for dim in hidden_dims:
            dim_str = str(dim)
            if dim_str in total_df.columns and baseline_method in total_df.index:
                baseline_time = total_df.loc[baseline_method, dim_str]
                
                # Skip if baseline time is 0 or not available
                if baseline_time == 0:
                    continue
                
                # Calculate speedup for each method
                for method in methods:
                    if method in total_df.index:
                        method_time = total_df.loc[method, dim_str]
                        
                        # Skip if method time is 0 (indicating no data)
                        if method_time == 0:
                            continue
                        
                        # Calculate speedup (baseline / method)
                        # For the baseline itself, this will be 1.0
                        speedup = baseline_time / method_time
                        
                        # Add to plot data
                        plot_data.append({
                            'dataset': dataset,
                            'method': method,
                            'method_display': method_display_names[method],
                            'hidden_dim': dim,
                            'speedup': speedup,
                            'time': method_time
                        })

# Convert to DataFrame for easier plotting
plot_df = pd.DataFrame(plot_data)

# Calculate geometric mean speedup of f3s over other methods for each hidden dimension
f3s_speedups_by_dim = {dim: {'dgl': [], 'dfgnn_tiling': [], 'flashSparse': []} for dim in hidden_dims}
for dataset in datasets:
    for dim in hidden_dims:
        # Get f3s time for this dataset and dimension
        f3s_data = plot_df[(plot_df['dataset'] == dataset) & 
                           (plot_df['hidden_dim'] == dim) & 
                           (plot_df['method'] == 'f3s')]
        
        if not f3s_data.empty:
            f3s_time = f3s_data['time'].values[0]
            
            # Calculate speedup over each other method
            for other_method in ['dgl', 'dfgnn_tiling', 'flashSparse']:
                if other_method != 'f3s':
                    other_data = plot_df[(plot_df['dataset'] == dataset) & 
                                        (plot_df['hidden_dim'] == dim) & 
                                        (plot_df['method'] == other_method)]
                    
                    if not other_data.empty:
                        other_time = other_data['time'].values[0]
                        if other_time > 0 and f3s_time > 0:
                            # Speedup of f3s over other method
                            speedup = other_time / f3s_time
                            f3s_speedups_by_dim[dim][other_method].append(speedup)

# Calculate geometric means for each dimension
geo_means_by_dim = {dim: {} for dim in hidden_dims}
for dim in hidden_dims:
    for method, speedups in f3s_speedups_by_dim[dim].items():
        if speedups:
            geo_means_by_dim[dim][method] = np.exp(np.mean(np.log(speedups)))
        else:
            geo_means_by_dim[dim][method] = 0

# Print geometric means by dimension
print("\nGeometric Mean Speedup of f3s over other methods by hidden dimension:")
for dim in hidden_dims:
    print(f"\nHidden dimension {dim}:")
    for method, geo_mean in geo_means_by_dim[dim].items():
        print(f"f3s vs {method_display_names.get(method, method)}: {geo_mean:.2f}x")

# Calculate overall geometric means (across all dimensions)
f3s_speedups = {'dgl': [], 'dfgnn_tiling': [], 'flashSparse': []}
for dim in hidden_dims:
    for method in f3s_speedups.keys():
        f3s_speedups[method].extend(f3s_speedups_by_dim[dim][method])

# Calculate overall geometric means
geo_means = {}
for method, speedups in f3s_speedups.items():
    if speedups:
        geo_means[method] = np.exp(np.mean(np.log(speedups)))
    else:
        geo_means[method] = 0

print("\nOverall Geometric Mean Speedup of f3s over other methods:")
for method, geo_mean in geo_means.items():
    print(f"f3s vs {method_display_names.get(method, method)}: {geo_mean:.2f}x")

# Increase font sizes
plt.rcParams.update({'font.size': 14})  # Increase base font size

# Create the plot
plt.figure(figsize=(25, 5))

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

current_pos = 0.1  # Further reduced to start bars closer to the left edge
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
                    else:
                        plt.bar(pos, speedup, bar_width, color=color,
                               label=display_name if d_idx == 0 and h_idx == 0 else "")
                    
                    # Add the actual time value as text on the bar, rounded to 1 decimal place
                    time_value = method_data['time'].values[0]
                    plt.text(pos, speedup + 0.05, f"{time_value:.0f}", 
                            ha='center', va='bottom', fontsize=13, rotation=90)
            
            # Add tick for this dimension
            x_ticks.append(current_pos + (n_methods * bar_width) / 2 - bar_width/2)
            x_tick_labels.append(f"{dim}")
            
            dataset_label_pos.append(current_pos + (n_methods * bar_width) / 2 - bar_width/2)
            
            # Move to next group
            current_pos += n_methods * bar_width + bar_width * 0.5  # Reduced spacing between dimension groups
        
    # Add dataset label in the middle of its groups
    if dataset_label_pos:
        mid_pos = sum(dataset_label_pos) / len(dataset_label_pos)
        plt.text(mid_pos, -0.25, dataset_display_names[dataset], ha='center', va='top', 
                fontsize=18, fontweight='bold')  # Increased font size and moved closer to x-axis
    
    # Add space between datasets (reduced)
    current_pos += bar_width * 1.0  # Reduced from 2.0

# Set x-axis ticks and labels
plt.xticks(x_ticks, x_tick_labels, fontsize=16)

# Add a prominent horizontal line at y=1 for reference
plt.axhline(y=1.0, color='black', linestyle='dotted', alpha=0.8, linewidth=1.5, zorder=5, dashes=(2, 3))

# Create custom legend handles to ensure all methods are shown
legend_handles = []
for method in methods:
    legend_handles.append(Patch(color=method_colors[method], label=method_display_names[method]))

# Add legend and labels (no title)
if gpu == 'A30':
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=4, fontsize=18)
plt.ylabel('Speedup over DF-GNN', fontsize=18)
plt.grid(axis='y', linestyle='-', alpha=0.9, color='#CCCCCC', linewidth=0.8)  # Enhanced grid visibility with thicker, more opaque lines

# Set explicit x-axis limits to reduce whitespace
min_x = min(x_positions) - bar_width
max_x = max(x_positions) + bar_width
plt.xlim(min_x, max_x)

# Ensure y-axis starts at 0 and ends at 3
if data_dir == '../gt_time_csv_H100':
  plt.ylim(bottom=0, top=3.5)
else:
  plt.ylim(bottom=0, top=2.7)

plt.yticks(fontsize=18)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, left=0.01, right=0.99, top=0.95)  # Further reduced left margin
plt.savefig(data_dir + '/graph_transformer_speedup_' + gpu + '.png', dpi=300, bbox_inches='tight')
plt.close()

print("Chart created and saved as 'graph_transformer_speedup_" + gpu + ".png'") 