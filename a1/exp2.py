import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")

for col in ["H_Score", "Economy"]:
    s = list(df[col])
    n = len(s)
    
    # 1. Manual sort
    for i in range(n):
        for j in range(i + 1, n):
            if s[j] < s[i]:
                s[i], s[j] = s[j], s[i]
                
    # 2. Manual Min, Q1, Median, Q3, Max
    mid = n // 2
    med = (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]
    low = s[:mid]
    high = s[mid:] if n % 2 == 0 else s[mid + 1:]
    
    q1 = (low[len(low)//2 - 1] + low[len(low)//2]) / 2 if len(low) % 2 == 0 else low[len(low)//2]
    q3 = (high[len(high)//2 - 1] + high[len(high)//2]) / 2 if len(high) % 2 == 0 else high[len(high)//2]
    mn, mx = s[0], s[-1]
    
    # 3. Manual IQR and outliers
    iqr = q3 - q1
    lf = q1 - 1.5 * iqr
    uf = q3 + 1.5 * iqr
    outliers = [x for x in s if x < lf or x > uf]
    
    print(f"{col} -> Min={mn}, Q1={q1}, Median={med}, Q3={q3}, Max={mx}")
    print(f"Fences: ({lf:.4f}, {uf:.4f}) | Outliers: {outliers}\n")

plt.boxplot([df["H_Score"], df["Economy"]], tick_labels=["H_Score", "Economy"])
plt.title("Box Plots of H_Score and Economy")
plt.show()