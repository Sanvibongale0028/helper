import pandas as pd

# Load dataset
df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")

for col in ["Fam", "Health"]:
    vals = list(df[col])
    n = len(vals)
    
    # Manual Min, Max, Mean, and Standard Deviation
    v_min = min(vals)
    v_max = max(vals)
    mean = sum(vals) / n
    std = (sum((x - mean) ** 2 for x in vals) / n) ** 0.5  # Population std (use n-1 for sample)
    
    # Min-Max Normalisation to [0, 1]
    minmax_vals = [(x - v_min) / (v_max - v_min) for x in vals]
    
    # Z-Score Normalisation
    zscore_vals = [(x - mean) / std for x in vals]
    
    # Side-by-side comparison for first 10 values
    comp = pd.DataFrame({
        f"{col}_Original": vals[:10],
        f"{col}_MinMax": minmax_vals[:10],
        f"{col}_ZScore": zscore_vals[:10]
    })
    print(f"=== {col}: First 10 Values ===")
    print(comp.to_string(index=False), "\n")
    
    # Verify Mean, Std, Min, Max after each method
    for name, norm_list in [("Min-Max", minmax_vals), ("Z-Score", zscore_vals)]:
        n_mean = sum(norm_list) / n
        n_std = (sum((x - n_mean) ** 2 for x in norm_list) / n) ** 0.5
        print(f"{col} ({name}) -> Mean: {n_mean:.4f}, Std: {n_std:.4f}, Min: {min(norm_list):.4f}, Max: {max(norm_list):.4f}")
    print("\n")