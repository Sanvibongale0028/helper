import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = [float(str(x).replace("%", "")) for x in df["Cocoa Percent"]]

def five_num_summary(vals):
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    med = (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]
    low = s[:mid]
    high = s[mid:] if n % 2 == 0 else s[mid + 1:]
    q1 = (low[len(low)//2 - 1] + low[len(low)//2]) / 2 if len(low) % 2 == 0 else low[len(low)//2]
    q3 = (high[len(high)//2 - 1] + high[len(high)//2]) / 2 if len(high) % 2 == 0 else high[len(high)//2]
    return s[0], q1, med, q3, s[-1]

for col in ["Rating", "Cocoa_Num"]:
    vals = list(df[col])
    mn, q1, med, q3, mx = five_num_summary(vals)
    iqr = q3 - q1
    lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = sorted([x for x in vals if x < lf or x > uf])
    print(f"=== {col} ===")
    print(f"Min: {mn}, Q1: {q1}, Median: {med}, Q3: {q3}, Max: {mx}")
    print(f"IQR Fences: [{lf:.3f}, {uf:.3f}] | Outlier Count: {len(outliers)}")
    print(f"Unique Outlier Values: {sorted(set(outliers))}\n")

# Box plots of Rating and Cocoa %
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.boxplot(df["Rating"])
plt.title("Box Plot of Rating")

plt.subplot(1, 2, 2)
plt.boxplot(df["Cocoa_Num"])
plt.title("Box Plot of Cocoa %")
plt.tight_layout()
plt.show()

# Side-by-side box plots of Rating for top 5 company locations
top5_locs = df["Company Location"].value_counts().head(5).index.tolist()
data_top5 = [df[df["Company Location"] == loc]["Rating"] for loc in top5_locs]

plt.figure(figsize=(8, 4))
plt.boxplot(data_top5, labels=top5_locs)
plt.title("Rating by Top 5 Company Locations")
plt.ylabel("Rating")
plt.show()