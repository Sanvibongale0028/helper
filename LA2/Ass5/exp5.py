
import pandas as pd

df = pd.read_csv("flavors_of_cacao.csv")
df.columns = df.columns.str.strip()

print("Original record count:", len(df))

print("\nMissing values per column BEFORE cleaning:")
print(df.isna().sum())

print("\nUnique values per column BEFORE cleaning:")
print(df.nunique(dropna=False))

# Standardise text columns
text_columns = df.select_dtypes(include=["object"]).columns.tolist()

for col in text_columns:
    df[col] = df[col].apply(
        lambda x: " ".join(str(x).strip().split()).title()
        if pd.notna(x) and str(x).strip() != ""
        else pd.NA
    )

# Convert Review Date to numeric
date_col = next(
    (c for c in df.columns if c.lower().strip() == "review date"),
    None
)

if date_col is None:
    raise ValueError("Review Date column not found.")

df[date_col] = pd.to_numeric(df[date_col], errors="coerce")

# Fill missing Review Date with its mode
date_mode = df[date_col].mode()

if len(date_mode) > 0:
    df[date_col] = df[date_col].fillna(date_mode.iloc[0])

# Convert Cocoa Percent to numeric
cocoa_col = next(
    (c for c in df.columns if c.lower().strip() == "cocoa percent"),
    None
)

if cocoa_col is None:
    raise ValueError("Cocoa Percent column not found.")

df[cocoa_col] = pd.to_numeric(
    df[cocoa_col].astype(str).str.replace("%", "", regex=False).str.strip(),
    errors="coerce"
)

# Count duplicate rows and remove them
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows found:", duplicate_count)

df = df.drop_duplicates().copy()

# Group company locations with fewer than 20 reviews into Other
location_col = next(
    (c for c in df.columns if c.lower().strip() == "company location"),
    None
)

if location_col is None:
    raise ValueError("Company Location column not found.")

location_counts = df[location_col].value_counts()
small_locations = location_counts[location_counts < 20].index

df[location_col] = df[location_col].apply(
    lambda x: "Other" if x in small_locations else x
)

print("\nRecord count AFTER cleaning:", len(df))

print("\nMissing values per column AFTER cleaning:")
print(df.isna().sum())

print("\nUnique values per column AFTER cleaning:")
print(df.nunique(dropna=False))

print("\nRecord count comparison")
print("Before cleaning:", pd.read_csv("flavors_of_cacao.csv").shape[0])
print("After cleaning:", len(df))

print("\nUnique location count after grouping:", df[location_col].nunique())
print("Final dataset preview:")
print(df.head())
