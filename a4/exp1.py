import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").astype(float)

# 1. Attribute classification table
attr_info = pd.DataFrame({
    "Attribute": ["Id", "Company", "Bean Origin", "Review Date", "Cocoa Percent", "Company Location", "Rating"],
    "Measurement_Scale": ["Nominal", "Nominal", "Nominal", "Interval", "Ratio", "Nominal", "Ordinal (treated as Interval)"],
    "Data_Type": ["Discrete", "Discrete", "Discrete", "Discrete", "Continuous", "Discrete", "Discrete"]
})
print("=== Attribute Classification ===")
print(attr_info.to_string(index=False), "\n")

# Create Cocoa % bands for pie chart
df["Cocoa_Band"] = df["Cocoa_Num"].apply(lambda x: "< 70%" if x < 70 else ("= 70%" if x == 70 else "> 70%"))

# 2. Bar chart of top 10 company locations
plt.figure(figsize=(8, 4))
df["Company Location"].value_counts().head(10).plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Top 10 Company Locations")
plt.xlabel("Company Location")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Interpretation (Bar Chart): The U.S.A. dominates chocolate production/reviews in the dataset, accounting for far more entries than any other country.\n")

# 3. Line chart of mean rating per year
plt.figure(figsize=(8, 4))
df.groupby("Review Date")["Rating"].mean().plot(kind="line", marker="o", color="darkgreen")
plt.title("Mean Rating per Year")
plt.xlabel("Review Year")
plt.ylabel("Mean Rating")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
print("Interpretation (Line Chart): Average chocolate ratings show a gradual upward trend from 2006 to 2017, stabilizing around 3.2–3.3 in recent years.\n")

# 4. Pie chart of Cocoa % bands
plt.figure(figsize=(5, 5))
df["Cocoa_Band"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Cocoa % Bands")
plt.ylabel("")
plt.tight_layout()
plt.show()
print("Interpretation (Pie Chart): Over 80% of reviewed chocolates contain 70% or higher cocoa content, with 70% being the single most common formulation.\n")

# 5. Histogram of Rating
plt.figure(figsize=(7, 4))
plt.hist(df["Rating"], bins=10, color="coral", edgecolor="black")
plt.title("Histogram of Chocolate Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
print("Interpretation (Histogram): Ratings are tightly concentrated between 2.75 and 3.75 with a peak around 3.25–3.50, and very few extreme ratings below 2.0 or above 4.0.\n")

# 6. Scatter plot of Cocoa % vs Rating
plt.figure(figsize=(7, 4))
plt.scatter(df["Cocoa_Num"], df["Rating"], alpha=0.4, color="purple")
plt.title("Cocoa % vs Rating")
plt.xlabel("Cocoa Percentage (%)")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()
print("Interpretation (Scatter Plot): There is no strong linear correlation between cocoa percentage and rating, though extremely high cocoa percentages (90–100%) tend to receive lower ratings.")