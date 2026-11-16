
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
year_col = find_column(["Review Date"])

df["Location"] = df[location_col].astype(str).str.strip().str.title()
df["Cocoa"] = pd.to_numeric(
    df[cocoa_col].astype(str).str.replace("%", "", regex=False),
    errors="coerce"
)
df["Rating"] = pd.to_numeric(df[rating_col], errors="coerce")
df["Year"] = pd.to_numeric(df[year_col], errors="coerce")

df = df.dropna(subset=["Location", "Cocoa", "Rating", "Year"]).copy()
df["Year"] = df["Year"].astype(int)
df = df[df["Year"].between(2006, 2017)].copy()

# Four items per transaction
top8 = df["Location"].value_counts().head(8).index.tolist()

def location_item(value):
    if value in top8:
        return "Location=" + value
    return "Location=Other"

def cocoa_band(value):
    if value < 70:
        return "Cocoa=Below70"
    elif value == 70:
        return "Cocoa=Exactly70"
    return "Cocoa=Above70"

def rating_class(value):
    if value < 3.0:
        return "Rating=Low"
    elif value < 3.5:
        return "Rating=Medium"
    return "Rating=High"

def period(year):
    if 2006 <= year <= 2009:
        return "Period=2006-09"
    elif 2010 <= year <= 2013:
        return "Period=2010-13"
    return "Period=2014-17"

transactions = []

for _, row in df.iterrows():
    transactions.append({
        location_item(row["Location"]),
        cocoa_band(row["Cocoa"]),
        rating_class(row["Rating"]),
        period(row["Year"])
    })

total = len(transactions)

def support_count(itemset):
    return sum(
        1 for transaction in transactions
        if itemset.issubset(transaction)
    )

def apriori(min_support):
    frequent = {}
    current = [
        {item}
        for item in sorted({
            item for transaction in transactions
            for item in transaction
        })
    ]

    k = 1
    counts_by_size = {}

    while current:
        frequent_current = []

        for candidate in current:
            count = support_count(candidate)
            sup = count / total if total else 0

            if sup >= min_support:
                frequent[frozenset(candidate)] = sup
                frequent_current.append(frozenset(candidate))

        if not frequent_current:
            break

        counts_by_size[k] = len(frequent_current)

        next_candidates = set()

        for a, b in combinations(frequent_current, 2):
            union = a | b

            if len(union) != k + 1:
                continue

            # All immediate subsets must be frequent
            valid = all(
                frozenset(subset) in frequent
                for subset in combinations(union, k)
            )

            if valid:
                next_candidates.add(union)

        current = [set(itemset) for itemset in next_candidates]
        k += 1

    return frequent, counts_by_size

thresholds = [0.02, 0.05, 0.10]
summary = []

for threshold in thresholds:
    frequent_itemsets, counts = apriori(threshold)

    summary.append({
        "Minimum Support": threshold,
        "L1 count": counts.get(1, 0),
        "L2 count": counts.get(2, 0),
        "L3 count": counts.get(3, 0),
        "L4 count": counts.get(4, 0),
        "Total frequent itemsets": sum(counts.values())
    })

    print("\nMinimum support:", threshold)
    for size in sorted(counts):
        print("L" + str(size), "frequent itemsets:", counts[size])

    print("Total frequent itemsets:", sum(counts.values()))

print("\nSUMMARY TABLE")
print(pd.DataFrame(summary).to_string(index=False))
