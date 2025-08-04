import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the CSV files
csv_dir = os.path.dirname(__file__)

# List of CSV files
csv_files = [
    "aggregate_resnet18_cifar100.csv",
    "aggregate_resnet18_fashion.csv",
    "aggregate_resnet18_tiny.csv",
    "aggregate_vgg16_cifar100.csv",
    "aggregate_vgg16_fashion.csv",
    "aggregate_vgg16_tiny.csv",
]

# Explicit color map
colors = {
    "resnet18_cifar100": "#1f77b4",
    "resnet18_fashion": "#ff7f0e",
    "resnet18_tiny": "#2ca02c",
    "vgg16_cifar100": "#d62728",
    "vgg16_fashion": "#9467bd",
    "vgg16_tiny": "#8c564b"
}

# Init plot
plt.figure(figsize=(12, 6))
line_count = 0

# Loop through files
for csv_file in csv_files:
    
    file_path = os.path.join(csv_dir, csv_file)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {csv_file}")
        continue

    df = pd.read_csv(file_path)

    # Filter to evaluation phase
    train_data = df[df['fase'] == 'train'].copy()


    # Ensure numeric and drop invalid
    train_data['epoca'] = pd.to_numeric(train_data['epoca'], errors='coerce')
    train_data['energy_consumed'] = pd.to_numeric(train_data['energy_consumed'], errors='coerce')
    train_data = train_data.dropna(subset=['epoca', 'energy_consumed'])

    # Skip first epoch (usually epoch==1)
    min_epoch = train_data['epoca'].min()
    train_data = train_data[train_data['epoca'] > min_epoch]

    # Skip if empty
    if train_data.empty or train_data['energy_consumed'].isnull().all():
        print(f"⚠️ Skipped empty or invalid: {csv_file}")
        continue

    # Check for duplicate epochs (diagnostic)
    dupes = train_data['epoca'].duplicated().any()
    if dupes:
        print(f"🚨 WARNING: Duplicate epochs found in {csv_file}")

    # Plot with fixed color
    label = csv_file.replace("aggregate_", "").replace(".csv", "")
    linestyles = {
        "resnet18_cifar100": "-",
        "resnet18_fashion": "--",
        "resnet18_tiny": "-.",
        "vgg16_cifar100": "-",
        "vgg16_fashion": "--",
        "vgg16_tiny": ":"
    }
    train_data = train_data.sort_values(by="epoca")

    plt.plot(
        train_data['epoca'], 
        train_data['energy_consumed'], 
        label=label, 
        color=colors[label], 
        linestyle=linestyles[label],
        linewidth=1.5
    )
    print(f"✅ Plotted {label} with color {colors[label]}")
    line_count += 1


# Plot final touches
plt.xlabel("Epoch")
plt.ylabel("Energy Consumed (kWh)")
plt.title("Energy Consumption per Epoch for Evaluation Phase")
plt.yscale('linear')
plt.legend()
plt.grid(True, which='both', axis='y')

# Debug: check actual colors
for line in plt.gca().get_lines():
    print(f"[DEBUG] Line '{line.get_label()}' uses color: {line.get_color()}")

# Final assert
assert line_count == 6, f"❌ Expected 6 lines, got {line_count}!"

# Save
output_path = os.path.join(csv_dir, "energy_consumption_train_plot.png")
plt.savefig(output_path)
print(f"✅ Plot saved to {output_path}")
