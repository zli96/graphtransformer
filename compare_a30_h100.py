import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tabulate import tabulate

# Directories containing the CSV files
a30_data_dir = 'gt_time_csv_A30'
h100_data_dir = 'gt_time_csv_H100'

# Get all CSV files in both directories
a30_csv_files = [f for f in os.listdir(a30_data_dir) if f.endswith('.csv') and 'total' in f]
h100_csv_files = [f for f in os.listdir(h100_data_dir) if f.endswith('.csv') and 'total' in f]

# Extract unique datasets
dataset_pattern = re.compile(r'gt_times_total_1heads_(.+)\.csv')
datasets = set()
for file in a30_csv_files:
    match = dataset_pattern.match(file)
    if match:
        datasets.add(match.group(1))

# Define custom order for datasets
custom_order = ['reddit', 'yelp', 'citeseer', 'pubmed', 'cora', 'igb_small', 'ogbn-arxiv', 'flickr', 'Questions', 'ogbn-products', 'cluster']
# Filter and order datasets according to custom order
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

# Create a dictionary to store the data for each hidden dimension
speedup_data = {dim: {method: {} for method in methods} for dim in hidden_dims}

# Load data and calculate speedups
for dataset in datasets:
    a30_file = f'gt_times_total_1heads_{dataset}.csv'
    h100_file = f'gt_times_total_1heads_{dataset}.csv'
    
    if a30_file in a30_csv_files and h100_file in h100_csv_files:
        a30_df = pd.read_csv(os.path.join(a30_data_dir, a30_file), index_col=0)
        h100_df = pd.read_csv(os.path.join(h100_data_dir, h100_file), index_col=0)
        
        # For each hidden dimension and method, calculate speedup
        for dim in hidden_dims:
            dim_str = str(dim)
            for method in methods:
                if method in a30_df.index and method in h100_df.index:
                    a30_time = a30_df.loc[method, dim_str]
                    h100_time = h100_df.loc[method, dim_str]
                    
                    # Skip if either time is 0 (indicating no data)
                    if a30_time == 0 or h100_time == 0:
                        speedup_data[dim][method][dataset] = float('nan')
                    else:
                        # Calculate speedup (A30 / H100)
                        speedup = a30_time / h100_time
                        speedup_data[dim][method][dataset] = speedup

# Create and print tables for each hidden dimension
for dim in hidden_dims:
    print(f"\n\nHidden Dimension: {dim}")
    
    # Create a DataFrame for this hidden dimension
    table_data = []
    for method in methods:
        row = [method_display_names[method]]
        for dataset in datasets:
            if dataset in speedup_data[dim][method]:
                speedup = speedup_data[dim][method][dataset]
                if np.isnan(speedup):
                    row.append("N/A")
                else:
                    row.append(f"{speedup:.2f}x")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Create DataFrame with proper headers
    headers = ["Method"] + [dataset.replace('_', ' ').title() for dataset in datasets]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)

# Calculate geometric means for each method and hidden dimension
print("\n\nGeometric Mean Speedups (H100 over A30):")
for dim in hidden_dims:
    print(f"\nHidden Dimension: {dim}")
    for method in methods:
        # Collect all valid speedups for this method and dimension
        speedups = []
        for dataset in datasets:
            if dataset in speedup_data[dim][method]:
                speedup = speedup_data[dim][method][dataset]
                if not np.isnan(speedup):
                    speedups.append(speedup)
        
        # Calculate geometric mean if we have data
        if speedups:
            geo_mean = np.exp(np.mean(np.log(speedups)))
            print(f"{method_display_names[method]}: {geo_mean:.2f}x")
        else:
            print(f"{method_display_names[method]}: N/A")

# Create a bar chart showing the geometric mean speedups for each method and hidden dimension
plt.figure(figsize=(12, 8))

# Set up the bar positions
bar_width = 0.2
index = np.arange(len(methods))

# Calculate geometric means for plotting
geo_means = {dim: [] for dim in hidden_dims}
for dim in hidden_dims:
    for method in methods:
        speedups = []
        for dataset in datasets:
            if dataset in speedup_data[dim][method]:
                speedup = speedup_data[dim][method][dataset]
                if not np.isnan(speedup):
                    speedups.append(speedup)
        
        if speedups:
            geo_mean = np.exp(np.mean(np.log(speedups)))
            geo_means[dim].append(geo_mean)
        else:
            geo_means[dim].append(0)

# Plot bars for each hidden dimension
for i, dim in enumerate(hidden_dims):
    plt.bar(index + i*bar_width, geo_means[dim], bar_width, 
            label=f'Hidden Dim {dim}', alpha=0.7)

# Add labels and legend
plt.xlabel('Method', fontsize=14)
plt.ylabel('Geometric Mean Speedup (H100 over A30)', fontsize=14)
plt.title('Speedup of H100 over A30 by Method and Hidden Dimension', fontsize=16)
plt.xticks(index + bar_width, [method_display_names[m] for m in methods], fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('h100_vs_a30_speedup.png', dpi=300)
plt.close()

print("\nBar chart saved as 'h100_vs_a30_speedup.png'") 