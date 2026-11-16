import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
factors = ["Economy", "Fam", "Health", "Freedom"]

for col in factors:
    s = list(df[col])
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
    mn, mx = s[0], s[-1]
    
    iqr = q3 - q1
    lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = [round(x, 4) for x in s if x < lf or x > uf]
    
    print(f"{col:<8} -> Min={mn:.4f}, Q1={q1:.4f}, Med={med:.4f}, Q3={q3:.4f}, Max={mx:.4f}, IQR={iqr:.4f}")
    print(f"Outliers: {outliers}\n")

plt.boxplot([df[c] for c in factors], tick_labels=factors)
plt.title("Side-by-Side Box Plots of GHI Factors")
plt.show()