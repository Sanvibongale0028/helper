import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = [float(str(x).replace("%", "")) for x in df["Cocoa Percent"]]

# 1. Manual function to sort and find Min, Q1, Median, Q3, Max
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

# 2. Compute summary and IQR outliers for Rating and Cocoa %
for col in ["Rating", "Cocoa_Num"]:
    vals = list(df[col])
    mn, q1, med, q3, mx = five_num(vals)
    iqr = q3 - q1
    lf = q1 - 1.5 * iqr
    uf = q3 + 1.5 * iqr
    outliers = [x for x in vals if x < lf or x > uf]
    
    print(f"{col}: Min={mn}, Q1={q1}, Median={med}, Q3={q3}, Max={mx}")
    print(f"Fences: ({lf:.2f}, {uf:.2f}) | Outliers ({len(outliers)}): {sorted(set(outliers))}\n")

# 3. Box plots of Rating and Cocoa %
plt.boxplot(df["Rating"])
plt.title("Box Plot of Rating")
plt.show()

plt.boxplot(df["Cocoa_Num"])
plt.title("Box Plot of Cocoa %")
plt.show()

# 4. Side-by-side box plots for top 5 company locations
top5 = df["Company Location"].value_counts().head(5).index.tolist()
plt.boxplot([df[df["Company Location"] == loc]["Rating"] for loc in top5], labels=top5)
plt.title("Rating for Top 5 Company Locations")
plt.show()