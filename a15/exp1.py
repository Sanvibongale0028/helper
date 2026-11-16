import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").astype(float)

# 1. Dataset characteristics & attribute types
print(f"Dataset Shape (Rows, Columns): {df.shape}")
print(f"Total Elements (Size): {df.size}")
print("\nMissing Values per Column:\n", df.isnull().sum())

attr_table = pd.DataFrame({
    "Attribute": ["Id", "Company", "Bean Origin", "Review Date", "Cocoa Percent", "Company Location", "Rating"],
    "Data_Type": ["Nominal (Discrete)", "Nominal (Discrete)", "Nominal (Discrete)", "Interval (Discrete)", "Ratio (Continuous)", "Nominal (Discrete)", "Ordinal/Interval (Discrete)"],
    "Unique_Values": [df[c].nunique() for c in ["Id", "Company", "Bean Origin", "Review Date", "Cocoa Percent", "Company Location", "Rating"]]
})
print("\n=== Attribute Types ===")
print(attr_table.to_string(index=False), "\n")

# Create Rating class for pie chart
df["Rating_Class"] = df["Rating"].apply(lambda r: "High (>= 3.5)" if r >= 3.5 else ("Mid (2.75 - 3.49)" if r >= 2.75 else "Low (< 2.75)"))

# 2. Line chart of the number of reviews per year
plt.figure(figsize=(8, 4))
df["Review Date"].value_counts().sort_index().plot(kind="line", marker="o", color="steelblue")
plt.title("Number of Reviews per Year")
plt.xlabel("Review Year")
plt.ylabel("Number of Reviews")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 3. Bar chart of the 10 highest-rated locations (minimum 20 reviews)
loc_stats = df.groupby("Company Location").agg(count=("Rating", "count"), mean_rating=("Rating", "mean"))
top10_rated = loc_stats[loc_stats["count"] >= 20].sort_values(by="mean_rating", ascending=False).head(10)

plt.figure(figsize=(9, 4))
top10_rated["mean_rating"].plot(kind="bar", color="darkgreen", edgecolor="black")
plt.title("Top 10 Highest-Rated Company Locations (Min 20 Reviews)")
plt.xlabel("Company Location")
plt.ylabel("Mean Rating")
plt.ylim(3.0, 3.45)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Pie chart of Rating class
plt.figure(figsize=(5, 5))
df["Rating_Class"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Rating Classes")
plt.ylabel("")
plt.tight_layout()
plt.show()

# 5. Histogram of Cocoa %
plt.figure(figsize=(7, 4))
plt.hist(df["Cocoa_Num"], bins=12, color="coral", edgecolor="black")
plt.title("Histogram of Cocoa Percentage")
plt.xlabel("Cocoa %")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 6. Scatter plot of Review Date vs Mean Rating
yearly_mean = df.groupby("Review Date")["Rating"].mean().reset_index()
plt.figure(figsize=(7, 4))
plt.scatter(yearly_mean["Review Date"], yearly_mean["Rating"], color="purple", s=60)
plt.title("Review Date vs Mean Rating")
plt.xlabel("Review Year")
plt.ylabel("Mean Rating")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()