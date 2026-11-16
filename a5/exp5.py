import pandas as pd

# Load dataset
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv")

# Record count & unique values BEFORE cleaning
before_summary = pd.DataFrame({
    "Attribute": df.columns,
    "Missing_Before": df.isnull().sum().values,
    "Unique_Before": [df[col].nunique() for col in df.columns]
})
print(f"Record Count Before Cleaning: {len(df)}")
print(before_summary.to_string(index=False), "\n")

# 1. Fill missing Review Dates with the mode
mode_year = df["Review Date"].mode()[0]
df["Review Date"] = df["Review Date"].fillna(mode_year)

# 2. Convert Cocoa Percent to numeric
df["Cocoa Percent"] = df["Cocoa Percent"].astype(str).str.replace("%", "").str.strip().astype(float)

# 3. Trim whitespace and standardise case on text columns
text_cols = ["Company", "Bean Origin", "Company Location"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

# 4. Check for duplicate rows (excluding Id column)
dup_count = df.drop(columns=["Id"]).duplicated().sum()
print(f"Duplicate rows found (excluding Id): {dup_count}")
df = df.drop_duplicates(subset=df.columns.difference(["Id"])).copy()

# 5. Group Company Locations with fewer than 20 reviews into "Other"
loc_counts = df["Company Location"].value_counts()
rare_locations = loc_counts[loc_counts < 20].index
df["Company Location"] = df["Company Location"].apply(lambda x: "Other" if x in rare_locations else x)

# Record count & unique values AFTER cleaning
after_summary = pd.DataFrame({
    "Attribute": df.columns,
    "Missing_After": df.isnull().sum().values,
    "Unique_After": [df[col].nunique() for col in df.columns]
})
print(f"\nRecord Count After Cleaning: {len(df)}")
print(after_summary.to_string(index=False))