import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
cols = ["Economy", "Fam", "Health", "Freedom", "H_Score"]

# 1. Dataset characteristics & attribute types
print("Size (Total Elements):", df.size)
print("Dimensionality (Rows, Columns):", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("\nAttribute Types: All 5 attributes (Economy, Fam, Health, Freedom, H_Score) are Numeric (Ratio, Continuous).")
print("\nRanges (Min to Max):")
for c in cols:
    print(f"{c}: {df[c].min()} to {df[c].max()}")

# 2. Histograms of all 5 attributes
df[cols].hist(figsize=(10, 6), bins=15)
plt.tight_layout()
plt.show()

# 3. Scatter plots of each factor vs H_Score
factors = ["Economy", "Fam", "Health", "Freedom"]
plt.figure(figsize=(10, 6))
for i, f in enumerate(factors):
    plt.subplot(2, 2, i + 1)
    plt.scatter(df[f], df["H_Score"])
    plt.xlabel(f)
    plt.ylabel("H_Score")
    plt.title(f"{f} vs H_Score")
plt.tight_layout()
plt.show()

# 4. Create Happiness class (Low < 4.5, Medium 4.5 - 6.0, High > 6.0)
df["Class"] = df["H_Score"].apply(lambda x: "High" if x > 6.0 else ("Medium" if x >= 4.5 else "Low"))

# Bar chart of mean factors per Happiness class
df.groupby("Class")[factors].mean().plot(kind="bar", title="Mean Factors per Happiness Class")
plt.ylabel("Mean Score")
plt.xticks(rotation=0)
plt.show()

# Pie chart of class counts
df["Class"].value_counts().plot(kind="pie", autopct="%1.1f%%", title="Happiness Class Counts")
plt.ylabel("")
plt.show()