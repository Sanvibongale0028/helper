
import pandas as pd
from itertools import combinations

df = pd.read_csv("flavors_of_cacao.csv")
df.columns = df.columns.str.strip()

def find_column(names):
    for name in names:
        for col in df.columns:
            if col.lower().strip() == name.lower():
                return col

    for name in names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col

    raise ValueError("Column not found: " + str(names))

location_col = find_column(["Company Location"])
cocoa_col = find_column(["Cocoa Percent"])
rating_col = find_column(["Rating"])
year_col = find_column(["Review Date", "Review Year"])

# Clean and convert data
df["Location"] = df[location_col].astype(str).str.strip()
df["Cocoa Numeric"] = pd.to_numeric(
    df[cocoa_col].astype(str).str.replace("%", "", regex=False),
    errors="coerce"
)
df["Rating Numeric"] = pd.to_numeric(df[rating_col], errors="coerce")
df["Year Numeric"] = pd.to_numeric(df[year_col], errors="coerce")

df = df.dropna(
    subset=["Location", "Cocoa Numeric", "Rating Numeric", "Year Numeric"]
).copy()

df["Year Numeric"] = df["Year Numeric"].astype(int)

# Keep only the requested review periods
df = df[
    df["Year Numeric"].between(2006, 2017)
].copy()

# 1. Company location: top 8, others become Other
location_counts = df["Location"].value_counts()
top8_locations = location_counts.head(8).index.tolist()

df["Location Item"] = df["Location"].apply(
    lambda x: "Location=" + x
    if x in top8_locations
    else "Location=Other"
)

# 2. Cocoa bands using 70 as the boundary
def cocoa_band(value):
    if value < 70:
        return "Cocoa=Below70"
    elif value == 70:
        return "Cocoa=Exactly70"
    else:
        return "Cocoa=Above70"

df["Cocoa Item"] = df["Cocoa Numeric"].apply(cocoa_band)

# 3. Rating class
def rating_class(value):
    if value < 3.0:
        return "Rating=Low"
    elif value < 3.5:
        return "Rating=Medium"
    else:
        return "Rating=High"

df["Rating Item"] = df["Rating Numeric"].apply(rating_class)

# 4. Review period
def review_period(year):
    if 2006 <= year <= 2009:
        return "Period=2006-09"
    elif 2010 <= year <= 2013:
        return "Period=2010-13"
    elif 2014 <= year <= 2017:
        return "Period=2014-17"
    return None

df["Period Item"] = df["Year Numeric"].apply(review_period)
df = df.dropna(subset=["Period Item"]).copy()

# Create one transaction per review
transactions = []

for _, row in df.iterrows():
    transactions.append({
        row["Location Item"],
        row["Cocoa Item"],
        row["Rating Item"],
        row["Period Item"]
    })

total = len(transactions)
min_support = 0.05

def count_itemset(itemset):
    return sum(
        1 for transaction in transactions
        if itemset.issubset(transaction)
    )

def support(itemset):
    if total == 0:
        return 0
    return count_itemset(itemset) / total

print("Total transactions:", total)
print("Minimum support:", min_support)

# Apriori algorithm
all_items = sorted({
    item
    for transaction in transactions
    for item in transaction
})

frequent_by_size = {}
current = [{item} for item in all_items]
k = 1

while current:
    frequent_current = []

    for candidate in current:
        sup = support(candidate)

        if sup >= min_support:
            frequent_by_size[frozenset(candidate)] = sup
            frequent_current.append(frozenset(candidate))

    if not frequent_current:
        break

    print("\nFREQUENT", k, "-ITEMSETS")
    for itemset in sorted(
        frequent_current,
        key=lambda x: sorted(x)
    ):
        print(
            sorted(itemset),
            "| Support count:", count_itemset(set(itemset)),
            "| Support:", round(frequent_by_size[itemset], 4)
        )

    next_candidates = set()

    # Join frequent itemsets and prune using Apriori property
    for a, b in combinations(frequent_current, 2):
        union = a | b

        if len(union) != k + 1:
            continue

        all_subsets_frequent = True

        for subset in combinations(union, k):
            if frozenset(subset) not in frequent_by_size:
                all_subsets_frequent = False
                break

        if all_subsets_frequent:
            next_candidates.add(union)

    current = [set(x) for x in next_candidates]
    k += 1

# Verify three itemset supports by direct counting
print("\nDIRECT SUPPORT VERIFICATION")

verification_sets = [
    {"Cocoa=Below70", "Rating=High"},
    {"Location=Other", "Rating=High"},
    {"Cocoa=Below70", "Rating=High", "Period=2014-17"}
]

for itemset in verification_sets:
    direct_count = sum(
        1 for transaction in transactions
        if all(item in transaction for item in itemset)
    )

    direct_support = direct_count / total if total else 0

    print("\nItemset:", sorted(itemset))
    print("Direct count:", direct_count)
    print("Direct support:", round(direct_support, 4))
    print("Meets minimum support:", direct_support >= min_support)
