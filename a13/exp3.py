import pandas as pd

# Load dataset
df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")

for col in ["Economy", "H_Score"]:
    vals = list(df[col])
    n = len(vals)
    
    # Basic manual statistics
    v_min = min(vals)
    v_max = max(vals)
    mean = sum(vals) / n
    std = (sum((x - mean) ** 2 for x in vals) / n) ** 0.5
    
    # 1. Min-Max normalisation to [-1, 1]
    # Formula: ((x - min) / (max - min)) * (new_max - new_min) + new_min
    minmax_neg1_1 = [((x - v_min) / (v_max - v_min)) * 2 - 1 for x in vals]
    
    # 2. Z-score normalisation
    zscore = [(x - mean) / std for x in vals]
    
    # 3. Decimal scaling: x / 10^j where j is smallest integer such that max(|x_scaled|) < 1
    max_abs = max(abs(x) for x in vals)
    j = 0
    while max_abs / (10 ** j) >= 1:
        j += 1
    dec_scaled = [x / (10 ** j) for x in vals]
    
    # Helper to summarise stats manually
    def stats_row(method_name, arr):
        m = sum(arr) / len(arr)
        s = (sum((v - m) ** 2 for v in arr) / len(arr)) ** 0.5
        return [method_name, round(min(arr), 4), round(max(arr), 4), round(m, 4), round(s, 4)]
        
    comp_table = pd.DataFrame([
        stats_row("Original", vals),
        stats_row("Min-Max [-1, 1]", minmax_neg1_1),
        stats_row("Z-Score", zscore),
        stats_row(f"Decimal Scaling (j={j})", dec_scaled)
    ], columns=["Method", "Min", "Max", "Mean", "Std"])
    
    print(f"=== {col}: Normalisation Comparison ===")
    print(comp_table.to_string(index=False), "\n")

print("Choice for K-Means: Z-score normalisation is preferred before K-Means because it centers every feature at mean = 0 with equal variance (std = 1), preventing features with wider natural spreads from dominating Euclidean distance calculations while remaining less sensitive to extreme outliers than Min-Max scaling.")