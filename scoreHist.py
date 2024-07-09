import pandas as pd
import matplotlib.pyplot as plt

# Load JSONL data into a pandas DataFrame
df = pd.read_json('ga_output.jsonl', lines=True)

# Filter out only valid samples
df_valid = df[df['valid'] == True]
df_valid = df

# Define a mapping from numeric generation to descriptive labels
generation_labels = {0: 'Start', 1: 'Mid', 2: 'Final'}

# Filter data by generation number and plot histograms with shared y-axis
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
generations = df_valid['gen#'].unique()
for i, gen in enumerate(sorted(generations)):
    subset = df_valid[df_valid['gen#'] == gen]
    axes[i].hist(subset['score'], bins=20, alpha=0.7, color='blue')
    # Use the generation label from the mapping
    label = generation_labels.get(gen, f"Gen {gen}")  # Fallback to "Gen {gen}" if not found in the map
    axes[i].set_title(f'Histogram of Scores for {label} (Valid Samples)')
    axes[i].set_xlabel('Score')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
