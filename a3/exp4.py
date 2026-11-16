import pandas as pd

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
vals = [float(str(x).replace("%", "").strip()) for x in df["Cocoa Percent"]]
n = len(vals)

# 1. Manual sort
for i in range(n):
    for j in range(i + 1, n):
        if vals[j] < vals[i]:
            vals[i], vals[j] = vals[j], vals[i]

# 2. Create 4 equal-width bins
v_min, v_max = vals[0], vals[-1]
w = (v_max - v_min) / 4
edges = [v_min + i * w for i in range(5)]

bins = [[], [], [], []]
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

# 3. Smooth by bin means, bin medians, and bin boundaries
s_mean, s_med, s_bound = [], [], []
for b in bins:
    m = len(b)
    tot = 0.0
    for x in b:
        tot += x
    mean_v = tot / m
    med_v = (b[m // 2 - 1] + b[m // 2]) / 2 if m % 2 == 0 else b[m // 2]
    low_v, high_v = b[0], b[-1]
    
    for x in b:
        s_mean.append(round(mean_v, 2))
        s_med.append(round(med_v, 2))
        d_low = x - low_v if x >= low_v else low_v - x
        d_high = high_v - x if high_v >= x else x - high_v
        s_bound.append(low_v if d_low <= d_high else high_v)

print("\n15 Sample Records (Original | Mean | Median | Boundary):")
for k in range(15):
    idx = int(k * (n - 1) / 14)
    print(f"{vals[idx]} | {s_mean[idx]} | {s_med[idx]} | {s_bound[idx]}")