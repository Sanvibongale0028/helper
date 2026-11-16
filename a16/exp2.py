import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
factors = ["Economy", "Fam", "Health", "Freedom"]

# Manual five-number summary function
def five_num(data):
    s = list(data)
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            if s[j] < s[i]:
                s[i], s[j] = s[j], s[i]
                
    mid = n // 2
    med = (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]
    low = s[:mid]
    high = s[mid:] if n % 2 == 0 else s[mid + 1:]
    
    q1 = (low[len(low)//2 - 1] + low[len(low)//2]) / 2 if len(low) % 2 == 0 else low[len(low)//2]
    q3 = (high[len(high)//2 - 1] + high[len(high)//2]) / 2 if len(high) % 2 == 0 else high[len(high)//2]
    return s[0], q1, med, q3, s[-1]

for col in factors:
    vals = list(df[col])
    mn, q1, med, q3, mx = five_num(vals)
    iqr = q3 - q1
    lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = [round(x, 4) for x in vals if x < lf or x > uf]
    
    print(f"{col:<8} -> Min={mn:.4f}, Q1={q1:.4f}, Med={med:.4f}, Q3={q3:.4f}, Max={mx:.4f}, IQR={iqr:.4f}")
    print(f"Outliers: {outliers}\n")

plt.boxplot([df[c] for c in factors], tick_labels=factors)
plt.title("Side-by-Side Box Plots of GHI Factors")
plt.show()

print("Spread Comparison: Economy has the largest spread (IQR = 0.6143, Range = 1.6904), whereas Freedom has the tightest spread (IQR = 0.2208, Range = 0.6697).")