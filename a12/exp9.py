import math
import pandas as pd

df = pd.read_csv("weather_numeric - weather_numeric.csv")
y = list(df["play"])
N = len(y)

# Manual Entropy & Gini helper functions
def entropy(labels):
    t = len(labels)
    if t == 0:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    e = 0.0
    for v in counts.values():
        p = v / t
        e -= p * math.log2(p)
    return e

def gini(labels):
    t = len(labels)
    if t == 0:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    s = 0.0
    for v in counts.values():
        p = v / t
        s += p * p
    return 1.0 - s

base_ent = entropy(y)
base_gini = gini(y)
print(f"Base Entropy = {base_ent:.4f}, Base Gini = {base_gini:.4f}\n")

# Full working table for categorical attribute: outlook
print("=== Full Working Table for 'outlook' ===")
outlook_col = list(df["outlook"])
we_out, wg_out = 0.0, 0.0
for val in ["sunny", "overcast", "rainy"]:
    sub = [y[i] for i in range(N) if outlook_col[i] == val]
    yes_c = sum(1 for l in sub if l == "yes")
    no_c = sum(1 for l in sub if l == "no")
    e_v = entropy(sub)
    g_v = gini(sub)
    we_out += (len(sub) / N) * e_v
    wg_out += (len(sub) / N) * g_v
    print(f"outlook = {val:<8} | Total={len(sub)}, yes={yes_c}, no={no_c} | Entropy={e_v:.4f}, Gini={g_v:.4f}")

print(f"Weighted Entropy = {we_out:.4f} -> Info Gain = {base_ent - we_out:.4f}")
print(f"Weighted Gini    = {wg_out:.4f} -> Gini Gain = {base_gini - wg_out:.4f}\n")

# Categorical attribute: windy
windy_col = list(df["windy"])
we_win, wg_win = 0.0, 0.0
for val in [True, False]:
    sub = [y[i] for i in range(N) if windy_col[i] == val]
    we_win += (len(sub) / N) * entropy(sub)
    wg_win += (len(sub) / N) * gini(sub)

results = [
    ["outlook", "Categorical", round(base_ent - we_out, 4), round(wg_out, 4), round(base_gini - wg_out, 4)],
    ["windy", "Categorical", round(base_ent - we_win, 4), round(wg_win, 4), round(base_gini - wg_win, 4)]
]

# Numeric attributes: temperature and humidity (test all midpoints)
for col in ["temperature", "humidity"]:
    col_vals = list(df[col])
    u = []
    for x in col_vals:
        if x not in u:
            u.append(x)
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            if u[j] < u[i]:
                u[i], u[j] = u[j], u[i]
                
    best_ig, t_ig = -1.0, 0.0
    best_g, t_g = 999.0, 0.0
    for i in range(len(u) - 1):
        t = (u[i] + u[i + 1]) / 2
        left = [y[k] for k in range(N) if col_vals[k] <= t]
        right = [y[k] for k in range(N) if col_vals[k] > t]
        es = (len(left) / N) * entropy(left) + (len(right) / N) * entropy(right)
        gs = (len(left) / N) * gini(left) + (len(right) / N) * gini(right)
        if base_ent - es > best_ig:
            best_ig, t_ig = base_ent - es, t
        if gs < best_g:
            best_g, t_g = gs, t
    results.append([col, f"<= {t_ig}", round(best_ig, 4), round(best_g, 4), round(base_gini - best_g, 4)])

print("=== Summary Table of All Attributes ===")
print("Attribute   | Best Split  | Info Gain | Split Gini | Gini Gain")
for r in results:
    print(f"{r[0]:<11} | {r[1]:<11} | {r[2]:<9} | {r[3]:<10} | {r[4]}")

print("\nSplitting Attribute & Agreement: Both Information Gain (0.2467) and Gini Index (lowest split Gini = 0.3429, highest Gini Gain = 0.1163) agree on choosing 'outlook' as the root splitting attribute.")