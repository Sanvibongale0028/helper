import pandas as pd

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
vals = sorted([float(x) for x in df["Rating"]])
N = len(vals)

# 1. 3 Equal-Width Bins
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

# 2. 3 Equal-Frequency (Equal-Depth) Bins
size = N // 3
ef_bins = [vals[:size], vals[size:2 * size], vals[2 * size:]]

print("Bin Counts Comparison:")
print(f"Equal-Width Counts:     {[len(b) for b in ew_bins]}")
print(f"Equal-Frequency Counts: {[len(b) for b in ef_bins]}\n")

# Smoothing helper
def smooth_bins(bin_list):
    s_mean, s_med, s_bound = [], [], []
    for b in bin_list:
        n = len(b)
        mean_v = sum(b) / n
        med_v = (b[n // 2 - 1] + b[n // 2]) / 2 if n % 2 == 0 else b[n // 2]
        low_v, high_v = b[0], b[-1]
        for x in b:
            s_mean.append(round(mean_v, 3))
            s_med.append(round(med_v, 3))
            s_bound.append(low_v if abs(x - low_v) <= abs(x - high_v) else high_v)
    return s_mean, s_med, s_bound

ew_mean, ew_med, ew_bound = smooth_bins(ew_bins)
ef_mean, ef_med, ef_bound = smooth_bins(ef_bins)

sample_idx = [0, 100, 300, 600, 900, 1200, 1500, 1700, 1794]
comp = pd.DataFrame({
    "Original": [vals[i] for i in sample_idx],
    "EW_Mean": [ew_mean[i] for i in sample_idx],
    "EW_Median": [ew_med[i] for i in sample_idx],
    "EW_Bound": [ew_bound[i] for i in sample_idx],
    "EF_Mean": [ef_mean[i] for i in sample_idx],
    "EF_Median": [ef_med[i] for i in sample_idx],
    "EF_Bound": [ef_bound[i] for i in sample_idx],
})
print("Sample Smoothed Records:")
print(comp.to_string(index=False))
print("\nExplanation: Equal-frequency binning suits this tightly clustered attribute better because most ratings cluster around 2.75–3.50, causing equal-width binning to dump almost all records into the middle bin while leaving extreme bins nearly empty.")