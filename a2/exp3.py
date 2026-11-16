import pandas as pd

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")

for col in ["Fam", "Health"]:
    vals = list(df[col])
    n = len(vals)
    
    # 1. Manual min, max, and sum
    v_min = vals[0]
    v_max = vals[0]
    total = 0.0
    for x in vals:
        if x < v_min:
            v_min = x
        if x > v_max:
            v_max = x
        total += x
        
    # 2. Manual mean and std
    mean = total / n
    sq = 0.0
    for x in vals:
        sq += (x - mean) ** 2
    std = (sq / n) ** 0.5
    
    # 3. Normalisation formulas
    mm = [(x - v_min) / (v_max - v_min) for x in vals]
    zs = [(x - mean) / std for x in vals]
    
    print(f"=== {col} (First 10 Values) ===")
    print("Original | Min-Max | Z-Score")
    for i in range(10):
        print(f"{vals[i]:.4f}   | {mm[i]:.4f}  | {zs[i]:.4f}")
        
    # 4. Verify Mean, Std, Min, Max after each method
    for name, arr in [("Min-Max", mm), ("Z-Score", zs)]:
        a_min, a_max, a_sum = arr[0], arr[0], 0.0
        for v in arr:
            if v < a_min: a_min = v
            if v > a_max: a_max = v
            a_sum += v
        a_mean = a_sum / n
        a_sq = 0.0
        for v in arr:
            a_sq += (v - a_mean) ** 2
        a_std = (a_sq / n) ** 0.5
        print(f"{name} -> Mean: {a_mean:.4f}, Std: {a_std:.4f}, Min: {a_min:.4f}, Max: {a_max:.4f}")
    print()