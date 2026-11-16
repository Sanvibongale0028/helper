import pandas as pd

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")

# Concept hierarchy for H_Score using fixed cut-offs (6.5, 5.5, 4.5)
def h_level(score):
    if score >= 6.5:
        return "Very High"
    elif score >= 5.5:
        return "High"
    elif score >= 4.5:
        return "Medium"
    else:
        return "Low"

df["H_Score_Level"] = df["H_Score"].apply(h_level)

# Concept hierarchy for Economy using tertiles (Rich / Middle / Poor)
df["Economy_Level"] = pd.qcut(df["Economy"], q=3, labels=["Poor", "Middle", "Rich"])

# 1. Mean of each factor per H_Score level
factors = ["Economy", "Fam", "Health", "Freedom"]
order = ["Very High", "High", "Medium", "Low"]
print("=== Mean Factors per H_Score Level ===")
print(df.groupby("H_Score_Level")[factors].mean().reindex(order), "\n")

# 2. Economy level x H_Score level count table
print("=== Economy Level x H_Score Level Count Table ===")
ct = pd.crosstab(df["Economy_Level"], df["H_Score_Level"])[order]
print(ct, "\n")

print("Summary Insight: Higher Economy, Family, Health, and Freedom means directly align with higher H_Score tiers. Almost all 'Very High' and 'High' happiness countries belong to the 'Rich' economy tertile, whereas 'Low' happiness countries are heavily concentrated in the 'Poor' economy tertile.")