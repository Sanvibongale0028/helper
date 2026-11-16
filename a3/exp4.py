import pandas as pd

# Load Cocoa dataset and extract numeric Cocoa Percent
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
vals = sorted([float(str(x).replace("%", "").strip()) for x in df["Cocoa Percent"]])

# Create 4 equal-width bins
v_min, v_max = vals[0], vals[-1]
width = (v_max - v_min) / 4

bins = [[], [], [], []]
edges = [v_min + i * width for i in range(5)]

for x in vals:
    if x <= edges[1]:
        bins[0].append(x)
    elif x <= edges[2]:
        bins[1].append(x)
    elif x <= edges[3]:
        bins[2].append(x)
    else:
        bins[3].append(x)

print("Bin Ranges and Counts:")
for i in range(4):
    print(f"Bin {i+1} [{edges[i]:.2f}, {edges[i+1]:.2f}]: Count = {len(bins[i])}")

# Smooth by bin means, bin medians, and bin boundaries
smooth_mean, smooth_median, smooth_bound = [], [], []

for b in bins:
    n = len(b)
    b_mean = sum(b) / n
    b_med = (b[n // 2 - 1] + b[n // 2]) / 2 if n % 2 == 0 else b[n // 2]
    b_low, b_high = b[0], b[-1]
    
    for x in b:
        smooth_mean.append(round(b_mean, 2))
        smooth_median.append(round(b_med, 2))
        # Closest boundary
        smooth_bound.append(b_low if abs(x - b_low) <= abs(x - b_high) else b_high)

# Display 15 sample records spread across the dataset
indices = [int(i * (len(vals) - 1) / 14) for i in range(15)]
sample_df = pd.DataFrame({
    "Original": [vals[i] for i in indices],
    "Bin_Mean": [smooth_mean[i] for i in indices],
    "Bin_Median": [smooth_median[i] for i in indices],
    "Bin_Boundary": [smooth_bound[i] for i in indices]
})
print("\n15 Sample Records Under All Three Smoothing Methods:")
print(sample_df.to_string(index=False))