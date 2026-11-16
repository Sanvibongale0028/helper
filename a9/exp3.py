import pandas as pd
import matplotlib.pyplot as plt

# Load dataset and extract numeric Cocoa %
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = [float(str(x).replace("%", "").strip()) for x in df["Cocoa Percent"]]

# Manual Min-Max [0, 1] and Z-score normalisation
norm_data = {}
for col in ["Rating", "Cocoa_Num"]:
    vals = list(df[col])
    n = len(vals)
    
    v_min = min(vals)
    v_max = max(vals)
    mean = sum(vals) / n
    std = (sum((x - mean) ** 2 for x in vals) / n) ** 0.5
    
    minmax_vals = [(x - v_min) / (v_max - v_min) for x in vals]
    zscore_vals = [(x - mean) / std for x in vals]
    
    norm_data[col] = {
        "Original": vals,
        "Min-Max [0, 1]": minmax_vals,
        "Z-Score": zscore_vals
    }

# Plot histograms before and after normalisation (2 rows x 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(13, 7))

for row_idx, col in enumerate(["Rating", "Cocoa_Num"]):
    for col_idx, method in enumerate(["Original", "Min-Max [0, 1]", "Z-Score"]):
        ax = axes[row_idx, col_idx]
        ax.hist(norm_data[col][method], bins=15, color="steelblue", edgecolor="black")
        ax.set_title(f"{col} — {method}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

print("Explanation: Both Min-Max and Z-score normalisation are strictly linear transformations of the form y = a*x + b (with a > 0). Because every data point is shifted and scaled proportionally, the relative distances and order between values remain identical, preserving the exact shape (skewness and peaks) of the histogram while only changing the x-axis scale.")