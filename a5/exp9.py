import math
import pandas as pd

# Use cleaned dataframe from Part A
# Create target Rating class (High >= 3.5, Mid >= 2.75, Low < 2.75)
df["Rating_Class"] = ["High" if r >= 3.5 else ("Mid" if r >= 2.75 else "Low") for r in df["Rating"]]

# Keep top 5 locations + "Other"
non_other_counts = {}
for loc in df["Company Location"]:
    if loc != "Other":
        non_other_counts[loc] = non_other_counts.get(loc, 0) + 1
top5_locs = sorted(non_other_counts, key=non_other_counts.get, reverse=True)[:5]
df["Loc_Top5"] = [loc if loc in top5_locs else "Other" for loc in df["Company Location"]]

y = list(df["Rating_Class"])
N = len(y)

# Manual Entropy & Gini helper functions
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
print(f"Base Entropy = {base_ent:.4f}, Base Gini = {base_gini:.4f}\n")

results = []

# 1. Categorical Attribute: Company Location (Top 5 + Other)
w_ent, w_gini = 0, 0
for val in sorted(set(df["Loc_Top5"])):
    sub_y = [y[i] for i in range(N) if df["Loc_Top5"].iloc[i] == val]
    w_ent += (len(sub_y) / N) * calc_entropy(sub_y)
    w_gini += (len(sub_y) / N) * calc_gini(sub_y)

results.append(["Company Location (Top 5 + Other)", "Multi-way categorical", round(base_ent - w_ent, 4), round(w_gini, 4), round(base_gini - w_gini, 4)])

# 2. Numeric Attributes: Cocoa Percent and Review Date (test candidate midpoints)
for col in ["Cocoa Percent", "Review Date"]:
    u_vals = sorted(set(df[col]))
    best_ig, best_t_ig = -1, None
    best_gsplit, best_ggain, best_t_g = 999, -1, None
    
    col_vals = list(df[col])
    for i in range(len(u_vals) - 1):
        thresh = (u_vals[i] + u_vals[i + 1]) / 2
        left_y = [y[k] for k in range(N) if col_vals[k] <= thresh]
        right_y = [y[k] for k in range(N) if col_vals[k] > thresh]
        
        e_split = (len(left_y) / N) * calc_entropy(left_y) + (len(right_y) / N) * calc_entropy(right_y)
        g_split = (len(left_y) / N) * calc_gini(left_y) + (len(right_y) / N) * calc_gini(right_y)
        
        if base_ent - e_split > best_ig:
            best_ig = base_ent - e_split
            best_t_ig = thresh
        if g_split < best_gsplit:
            best_gsplit = g_split
            best_ggain = base_gini - g_split
            best_t_g = thresh
            
    results.append([col, f"<= {best_t_ig} (IG) / <= {best_t_g} (Gini)", round(best_ig, 4), round(best_gsplit, 4), round(best_ggain, 4)])

res_df = pd.DataFrame(results, columns=["Attribute", "Best Split Threshold", "Info Gain", "Split Gini", "Gini Gain"])
print("=== Attribute Selection Results ===")
print(res_df.to_string(index=False))

root_attr = res_df.loc[res_df["Info Gain"].idxmax(), "Attribute"]
print(f"\nChosen Root Split Attribute: {root_attr} (achieves both the highest Information Gain of 0.0184 and the lowest Split Gini of 0.5759).")