import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
factors = ["Economy", "Fam", "Health", "Freedom"]

# Manual five-number summary function
def five_number_summary(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    
    if n % 2 == 0:
        median = (s[mid - 1] + s[mid]) / 2
        lower = s[:mid]
        upper = s[mid:]
    else:
        median = s[mid]
        lower = s[:mid]
        upper = s[mid + 1:]
        
    q1 = (lower[len(lower)//2 - 1] + lower[len(lower)//2]) / 2 if len(lower) % 2 == 0 else lower[len(lower)//2]
    q3 = (upper[len(upper)//2 - 1] + upper[len(upper)//2]) / 2 if len(upper) % 2 == 0 else upper[len(upper)//2]
    
    return s[0], q1, median, q3, s[-1]

# Compute five-number summary, IQR, fences, and outliers for each factor
summary_rows = []
for col in factors:
    vals = list(df[col])
    mn, q1, med, q3, mx = five_number_summary(vals)
    iqr = q3 - q1
    lf = q1 - 1.5 * iqr
    uf = q3 + 1.5 * iqr
    outliers = [round(x, 5) for x in vals if x < lf or x > uf]
    
    summary_rows.append([col, round(mn, 4), round(q1, 4), round(med, 4), round(q3, 4), round(mx, 4), round(iqr, 4), outliers])

res_df = pd.DataFrame(summary_rows, columns=["Attribute", "Min", "Q1", "Median", "Q3", "Max", "IQR", "Outliers"])
print("=== Five-Number Summary & Outliers ===")
print(res_df.to_string(index=False), "\n")

# Draw side-by-side box plots
plt.figure(figsize=(8, 5))
plt.boxplot([df[col] for col in factors], labels=factors)
plt.title("Side-by-Side Box Plots of the Four GHI Factors")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

print("Spread Comparison: Economy has the widest overall range (0.00 to 1.69) and largest IQR (~0.61), indicating the highest dispersion across countries. Freedom has the narrowest IQR (~0.22), while Fam is the only factor with low-end statistical outliers (0.1400 and 0.3029).")