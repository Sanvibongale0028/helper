import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")
df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").astype(float)

# 1. Attribute classification
print("Id: Nominal, Discrete")
print("Company: Nominal, Discrete")
print("Bean Origin: Nominal, Discrete")
print("Review Date: Interval, Discrete")
print("Cocoa Percent: Ratio, Continuous")
print("Company Location: Nominal, Discrete")
print("Rating: Ordinal (Interval), Discrete\n")

# 2. Bar chart of top 10 company locations
df["Company Location"].value_counts().head(10).plot(kind="bar", title="Top 10 Company Locations")
plt.show()
print("Bar Chart: The U.S.A. accounts for the highest number of chocolate reviews.\n")

# 3. Line chart of mean rating per year
df.groupby("Review Date")["Rating"].mean().plot(kind="line", marker="o", title="Mean Rating per Year")
plt.show()
print("Line Chart: Mean ratings gradually increased from 2006 to 2017.\n")

# 4. Pie chart of Cocoa % bands
bands = df["Cocoa_Num"].apply(lambda x: "<70%" if x < 70 else ("=70%" if x == 70 else ">70%"))
bands.value_counts().plot(kind="pie", autopct="%1.1f%%", title="Cocoa % Bands")
plt.ylabel("")
plt.show()
print("Pie Chart: Over 80% of chocolates have 70% or higher cocoa content.\n")

# 5. Histogram of Rating
plt.hist(df["Rating"], bins=10)
plt.title("Histogram of Rating")
plt.show()
print("Histogram: Ratings are tightly clustered between 2.75 and 3.75.\n")

# 6. Scatter plot of Cocoa % vs Rating
plt.scatter(df["Cocoa_Num"], df["Rating"])
plt.xlabel("Cocoa %")
plt.ylabel("Rating")
plt.title("Cocoa % vs Rating")
plt.show()
print("Scatter Plot: There is no strong linear relationship, though extreme cocoa percentages (>90%) get lower ratings.")