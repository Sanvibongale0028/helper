import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = [float(str(x).replace("%", "")) for x in df["Cocoa Percent"]]

for col in ["Rating", "Cocoa_Num"]:
    vals = list(df[col])
    n = len(vals)
    
    # 1. Manual min, max, mean, and std
    v_min, v_max, total = vals[0], vals[0], 0.0
    for x in vals:
        if x < v_min: v_min = x
        if x > v_max: v_max = x
        total += x
        
    mean = total / n
    sq = 0.0
    for x in vals:
        sq += (x - mean) ** 2
    std = (sq / n) ** 0.5
    
    # 2. Normalisation formulas
    mm = [(x - v_min) / (v_max - v_min) for x in vals]
    zs = [(x - mean) / std for x in vals]
    
    # 3. Plot histograms before and after
    plt.figure(figsize=(12, 3))
    
    plt.subplot(1, 3, 1)
    plt.hist(vals, bins=15)
    plt.title(col + " - Original")
    
    plt.subplot(1, 3, 2)
    plt.hist(mm, bins=15)
    plt.title(col + " - Min-Max [0, 1]")
    
    plt.subplot(1, 3, 3)
    plt.hist(zs, bins=15)
    plt.title(col + " - Z-Score")
    
    plt.tight_layout()
    plt.show()

print("Why shape does not change: Both Min-Max and Z-score are linear transformations (y = a*x + b). Every value is shifted and scaled proportionally, so the relative spacing and distribution shape stay identical—only the x-axis scale changes.")