import pandas as pd

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
vals = list(df["Rating"])
n = len(vals)

# 1. Manual sort
for i in range(n):
    for j in range(i + 1, n):
        if vals[j] < vals[i]:
            vals[i], vals[j] = vals[j], vals[i]

# Helper to smooth a list of bins by mean, median, and boundary
def smooth_bins(bins):
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
    return s_mean, s_med, s_bound

# 2. Equal-Width Binning (3 bins)
v_min, v_max = vals[0], vals[-1]
w = (v_max - v_min) / 3
ew_bins = [[], [], []]
for x in vals:
    if x <= v_min + w:
        ew_bins[0].append(x)
    elif x <= v_min + 2 * w:
        ew_bins[1].append(x)
    else:
        ew_bins[2].append(x)

# 3. Equal-Frequency Binning (3 bins)
size = n // 3
ef_bins = [vals[:size], vals[size:2 * size], vals[2 * size:]]

print("Equal-Width Bin Counts    :", [len(b) for b in ew_bins])
print("Equal-Frequency Bin Counts:", [len(b) for b in ef_bins])

ew_mean, ew_med, ew_bound = smooth_bins(ew_bins)
ef_mean, ef_med, ef_bound = smooth_bins(ef_bins)

print("\nSample Smoothed Values (Original | EW Mean, Med, Bound | EF Mean, Med, Bound):")
for k in range(10):
    idx = int(k * (n - 1) / 9)
    print(f"{vals[idx]} | {ew_mean[idx]}, {ew_med[idx]}, {ew_bound[idx]} | {ef_mean[idx]}, {ef_med[idx]}, {ef_bound[idx]}")

print("\nSuitability Explanation: Equal-frequency binning suits this tightly clustered attribute better because most ratings lie between 2.75 and 3.75—equal-width binning puts over 1,400 records into the middle bin and leaves the outer bins nearly empty, whereas equal-frequency distributes records evenly across all three bins.")