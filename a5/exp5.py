import pandas as pd

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")

print("Missing values per column:\n", df.isnull().sum())
print("\nBefore -> Records:", len(df), "| Unique locations:", df["Company Location"].nunique())

# 1. Fill missing Review Dates with mode
df["Review Date"] = df["Review Date"].fillna(df["Review Date"].mode()[0])

# 2. Convert Cocoa Percent to number
df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").str.strip().astype(float)

# 3. Trim and standardise text columns
for col in ["Company", "Bean Origin", "Company Location"]:
    df[col] = df[col].str.strip().str.title()

# 4. Drop duplicate rows
df = df.drop_duplicates(subset=["Company", "Bean Origin", "Review Date", "Cocoa Percent", "Company Location", "Rating"]).copy()

# 5. Group locations with < 20 reviews into "Other"
counts = df["Company Location"].value_counts()
df["Company Location"] = df["Company Location"].apply(lambda x: "Other" if counts[x] < 20 else x)

print("After  -> Records:", len(df), "| Unique locations:", df["Company Location"].nunique())