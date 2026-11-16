import pandas as pd

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")

def calc_stats(arr):
    n = len(arr)
    mn, mx, tot = arr[0], arr[0], 0.0
    for x in arr:
        if x < mn: mn = x
        if x > mx: mx = x
        tot += x
    mean = tot / n
    sq = 0.0
    for x in arr:
        sq += (x - mean) ** 2
    std = (sq / n) ** 0.5
    return mn, mx, mean, std

for col in ["Economy", "H_Score"]:
    vals = list(df[col])
    v_min, v_max, mean, std = calc_stats(vals)
    
    # 1. Min-Max to [-1, 1]
    mm = [((x - v_min) / (v_max - v_min)) * 2 - 1 for x in vals]
    
    # 2. Z-Score
    zs = [(x - mean) / std for x in vals]
    
    # 3. Decimal Scaling
    max_abs = v_max if v_max >= -v_min else -v_min
    j = 0
    while max_abs / (10 ** j) >= 1:
        j += 1
    ds = [x / (10 ** j) for x in vals]
    
    print(f"=== {col} Comparison ===")
    print("Method          | Min     | Max    | Mean   | Std")
    for name, arr in [("Original", vals), ("Min-Max [-1, 1]", mm), ("Z-Score", zs), (f"Decimal (j={j})", ds)]:
        a_min, a_max, a_mean, a_std = calc_stats(arr)
        print(f"{name:<15} | {a_min:>7.4f} | {a_max:>6.4f} | {a_mean:>6.4f} | {a_std:>6.4f}")
    print()

print("Preferred before K-Means: Z-Score normalisation, because it centers all attributes at mean = 0 with equal standard deviation = 1, preventing wider-scale attributes from dominating Euclidean distance while being less distorted by outliers than Min-Max.")