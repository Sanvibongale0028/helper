import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")

# Dataset characteristics
print(f"Size (Total elements): {df.size}")
print(f"Dimensionality (Rows, Cols): {df.shape}")
print("\nAttribute Data Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nAttribute Ranges (Max - Min):\n", df.max() - df.min())

# Create Happiness classes for bar & pie charts
df["Happiness_Class"] = pd.cut(df["H_Score"], bins=[0, 4.5, 6.0, 10], labels=["Low", "Medium", "High"])

# 1. Histograms of all 5 attributes
df[["Economy", "Fam", "Health", "Freedom", "H_Score"]].hist(figsize=(10, 6), bins=15, edgecolor="black")
plt.suptitle("Histograms of GHI Attributes")
plt.tight_layout()
plt.show()

# 2. Scatter plots of each factor vs H_Score
factors = ["Economy", "Fam", "Health", "Freedom"]
plt.figure(figsize=(10, 6))
for i, f in enumerate(factors, 1):
    plt.subplot(2, 2, i)
    plt.scatter(df[f], df["H_Score"], alpha=0.7)
    plt.xlabel(f)
    plt.ylabel("H_Score")
    plt.title(f"{f} vs H_Score")
plt.tight_layout()
plt.show()

# 3. Bar chart of mean factors per Happiness class
df.groupby("Happiness_Class", observed=False)[factors].mean().plot(kind="bar", figsize=(8, 4))
plt.title("Mean Factors per Happiness Class")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.show()

# 4. Pie chart of class counts
df["Happiness_Class"].value_counts().plot(kind="pie", autopct="%1.1f%%", figsize=(5, 5), title="Happiness Class Counts")
plt.ylabel("")
plt.show()