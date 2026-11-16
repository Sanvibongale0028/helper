import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").astype(float)

# 1. Bar chart of number of reviews per year
df["Review Date"].value_counts().sort_index().plot(kind="bar", title="Reviews per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# 2. Line chart of mean Rating per year
df.groupby("Review Date")["Rating"].mean().plot(kind="line", marker="o", title="Mean Rating per Year")
plt.ylabel("Mean Rating")
plt.show()

# 3. Pie chart of top 6 company locations vs Other
top6 = df["Company Location"].value_counts().head(6).index
loc_grp = df["Company Location"].apply(lambda x: x if x in top6 else "Other")
loc_grp.value_counts().plot(kind="pie", autopct="%1.1f%%", title="Top 6 Locations vs Other")
plt.ylabel("")
plt.show()

# 4. Histogram of Rating
plt.hist(df["Rating"], bins=10)
plt.title("Histogram of Rating")
plt.xlabel("Rating")
plt.show()

# 5. Scatter plot of Cocoa % vs Rating
plt.scatter(df["Cocoa_Num"], df["Rating"])
plt.title("Cocoa % vs Rating")
plt.xlabel("Cocoa %")
plt.ylabel("Rating")
plt.show()