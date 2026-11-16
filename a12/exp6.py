import pandas as pd

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
factors = ["Economy", "Fam", "Health", "Freedom"]

# 1. Concept hierarchy for H_Score using fixed cut-offs (6.5, 5.5, 4.5)
df["H_Level"] = df["H_Score"].apply(
    lambda x: "Very High" if x > 6.5 else ("High" if x > 5.5 else ("Medium" if x > 4.5 else "Low"))
)

# 2. Concept hierarchy for Economy by tertiles (equal frequency)
df["Econ_Level"] = pd.qcut(df["Economy"], q=3, labels=["Poor", "Middle", "Rich"])

# 3. Mean of each factor per H_Score level
print("=== Mean of Each Factor per H_Score Level ===")
print(df.groupby("H_Level", observed=False)[factors].mean().round(4), "\n")

# 4. Economy level x H_Score level count table
print("=== Economy Level x H_Score Level Count Table ===")
print(pd.crosstab(df["Econ_Level"], df["H_Level"]))

print("\nComment: There is a strong positive relationship between economic strength and happiness—26 of the 31 'Very High' happiness countries belong to the 'Rich' economy group and zero are 'Poor', whereas 27 of the 35 'Low' happiness countries fall in the 'Poor' economy group.")