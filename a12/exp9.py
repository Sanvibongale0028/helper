import pandas as pd
import math

df = pd.read_csv("weather_numeric - weather_numeric.csv")
y = list(df["play"])
N = len(y)

# Manual Entropy and Gini functions
def calc_entropy(labels):
    total = len(labels)
    if total == 0:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def calc_gini(labels):
    total = len(labels)
    if total == 0:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return 1 - sum((c / total) ** 2 for c in counts.values())

base_ent = calc_entropy(y)
base_gini = calc_gini(y)
print(f"Dataset Base Entropy = {base_ent:.4f}, Base Gini = {base_gini:.4f}\n")

results = []

# 1. Categorical attributes: outlook, windy (with full working table for outlook)
for col in ["outlook", "windy"]:
    vals = sorted(set(df[col]))
    w_ent, w_gini = 0, 0
    if col == "outlook":
        print("=== Full Working Table for Attribute: outlook ===")
        print(f"{'Value':<10} {'Total':<6} {'Yes':<5} {'No':<5} {'Entropy':<10} {'Gini':<10}")
    for v in vals:
        sub_y = [y[i] for i in range(N) if df[col].iloc[i] == v]
        e = calc_entropy(sub_y)
        g = calc_gini(sub_y)
        w_ent += (len(sub_y) / N) * e
        w_gini += (len(sub_y) / N) * g
        if col == "outlook":
            print(f"{str(v):<10} {len(sub_y):<6} {sub_y.count('yes'):<5} {sub_y.count('no'):<5} {e:<10.4f} {g:<10.4f}")
    if col == "outlook":
        print()
    results.append([col, "All categories", base_ent - w_ent, w_gini, base_gini - w_gini])

# 2. Numeric attributes: temperature, humidity (test all consecutive midpoints)
for col in ["temperature", "humidity"]:
    u_vals = sorted(set(df[col]))
    best_ig, best_t_ig = -1, None
    best_gsplit, best_ggain, best_t_g = 999, -1, None
    
    for i in range(len(u_vals) - 1):
        thresh = (u_vals[i] + u_vals[i + 1]) / 2
        left_y = [y[k] for k in range(N) if df[col].iloc[k] <= thresh]
        right_y = [y[k] for k in range(N) if df[col].iloc[k] > thresh]
        
        w_e = (len(left_y) / N) * calc_entropy(left_y) + (len(right_y) / N) * calc_entropy(right_y)
        w_g = (len(left_y) / N) * calc_gini(left_y) + (len(right_y) / N) * calc_gini(right_y)
        
        if base_ent - w_e > best_ig:
            best_ig = base_ent - w_e
            best_t_ig = thresh
        if w_g < best_gsplit:
            best_gsplit = w_g
            best_ggain = base_gini - w_g
            best_t_g = thresh
            
    results.append([col, f"<= {best_t_ig} (IG) / <= {best_t_g} (Gini)", best_ig, best_gsplit, best_ggain])

res_df = pd.DataFrame(results, columns=["Attribute", "Best Split", "Info Gain", "Split Gini", "Gini Gain"])
print("=== Tabulated Attribute Selection Results ===")
print(res_df.to_string(index=False))

best_ig_attr = res_df.loc[res_df["Info Gain"].idxmax(), "Attribute"]
best_gini_attr = res_df.loc[res_df["Split Gini"].idxmin(), "Attribute"]
print(f"\nSelected by Information Gain: {best_ig_attr}")
print(f"Selected by Gini Index: {best_gini_attr}")
print("Do Information Gain and Gini agree?", "Yes" if best_ig_attr == best_gini_attr else "No")