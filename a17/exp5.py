import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")

# 1. Top 10 Bean Origin values BEFORE cleaning
print("=== Top 10 Bean Origin BEFORE Cleaning ===")
print(df["Bean Origin"].value_counts().head(10), "\n")

# Clean Bean Origin (trim spaces and standardise case)
df["Bean Origin"] = df["Bean Origin"].astype(str).str.strip().str.title()

print("=== Top 10 Bean Origin AFTER Cleaning ===")
print(df["Bean Origin"].value_counts().head(10), "\n")

# 2. Handle missing Review Dates (impute with median year)
df["Review Date"] = df["Review Date"].fillna(df["Review Date"].median()).astype(int)

# 3. Convert Cocoa Percent to numeric and smooth by bin means (5 equal-width bins)
df["Cocoa_Num"] = df["Cocoa Percent"].astype(str).str.replace("%", "").str.strip().astype(float)
df["Cocoa_Bin"] = pd.cut(df["Cocoa_Num"], bins=5)
df["Cocoa_Smoothed"] = df.groupby("Cocoa_Bin", observed=False)["Cocoa_Num"].transform("mean").round(2)

# 4. Min-Max normalise Rating to [0, 1]
scaler = MinMaxScaler()
df["Rating_Norm"] = scaler.fit_transform(df[["Rating"]]).round(4)

# 5. Drop attributes not useful for mining and justify each
# - 'Id': Purely a sequential surrogate identifier with no predictive or pattern value.
# - 'Cocoa Percent' & 'Cocoa_Bin': Redundant raw string / intermediate bin columns replaced by 'Cocoa_Smoothed'.
df_cleaned = df.drop(columns=["Id", "Cocoa Percent", "Cocoa_Bin"])

print("Dropped Columns & Justifications:")
print("- Id: Sequential row identifier with zero predictive or clustering value.")
print("- Cocoa Percent & Cocoa_Bin: Replaced by cleaned numeric 'Cocoa_Smoothed' attribute.\n")

print("=== Preview of Preprocessed Dataset ===")
print(df_cleaned.head(10).to_string(index=False))