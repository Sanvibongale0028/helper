
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# ============================================================
# PART A: DATA TYPES AND VISUALIZATION
# ============================================================

FILE = "flavors_of_cacao.csv"

df = pd.read_csv(FILE)
df.columns = [str(c).strip() for c in df.columns]

print("PART A: DATASET INFORMATION")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("Columns:", list(df.columns))


# Find columns despite slight differences in their names
def find_column(possible_names, columns):
    for name in possible_names:
        for column in columns:
            if column.lower().strip() == name.lower().strip():
                return column

    raise ValueError(
        "Column not found. Available columns: " + str(columns)
    )


company_col = find_column(
    ["Company (Maker-if known)", "Company"], list(df.columns)
)

location_col = find_column(
    ["Company Location"], list(df.columns)
)

year_col = find_column(
    ["Review Date"], list(df.columns)
)

cocoa_col = find_column(
    ["Cocoa Percent"], list(df.columns)
)

rating_col = find_column(
    ["Rating"], list(df.columns)
)


# Convert required columns into ordinary Python lists
locations = df[location_col].fillna("").astype(str).tolist()
years_raw = df[year_col].tolist()
cocoa_raw = df[cocoa_col].fillna("").astype(str).tolist()
ratings_raw = df[rating_col].tolist()

years = []
cocoa = []
ratings = []

# Clean and convert data manually
for i in range(len(df)):
    try:
        year = int(float(years_raw[i]))
        rating = float(ratings_raw[i])
        cocoa_value = float(
            cocoa_raw[i].replace("%", "").strip()
        )

        if 0 <= cocoa_value <= 100 and 0 <= rating <= 5:
            years.append(year)
            cocoa.append(cocoa_value)
            ratings.append(rating)
        else:
            years.append(None)
            cocoa.append(None)
            ratings.append(None)

    except (ValueError, TypeError):
        years.append(None)
        cocoa.append(None)
        ratings.append(None)


print("\nATTRIBUTE TYPES")
print("Company: Nominal")
print("Company Location: Nominal")
print("Bean Origin: Nominal")
print("Review Date (year): Interval")
print("Cocoa Percent: Ratio")
print("Rating: Ordinal")
print("Bean Type: Nominal")
print("REF: Identifier / nominal")


# 1. Bar chart: Top 10 company locations by review count
location_counts = {}

for location in locations:
    location = location.strip()

    if location:
        location_counts[location] = (
            location_counts.get(location, 0) + 1
        )

top_locations = sorted(
    location_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nTOP 10 COMPANY LOCATIONS")

for location, count in top_locations:
    print(location, count)

plt.figure()
plt.bar(
    [x[0] for x in top_locations],
    [x[1] for x in top_locations]
)
plt.xlabel("Company Location")
plt.ylabel("Number of Reviews")
plt.title("Top 10 Company Locations")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("Interpretation: The tallest bar represents the location with the most reviews.")


# 2. Line chart: Mean rating per year
year_sum = {}
year_count = {}

for i in range(len(df)):
    if years[i] is not None and ratings[i] is not None:
        year = years[i]
        year_sum[year] = year_sum.get(year, 0) + ratings[i]
        year_count[year] = year_count.get(year, 0) + 1

sorted_years = sorted(year_sum.keys())
mean_ratings = []

for year in sorted_years:
    mean_ratings.append(year_sum[year] / year_count[year])

plt.figure()
plt.plot(sorted_years, mean_ratings, marker="o")
plt.xlabel("Review Year")
plt.ylabel("Mean Rating")
plt.title("Mean Chocolate Rating per Year")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Interpretation: The line shows how average chocolate ratings change over the years.")


# 3. Pie chart: Cocoa percentage bands
bands = {
    "Below 60%": 0,
    "60-69%": 0,
    "70-79%": 0,
    "80-89%": 0,
    "90% and above": 0
}

for value in cocoa:
    if value is None:
        continue
    elif value < 60:
        bands["Below 60%"] += 1
    elif value < 70:
        bands["60-69%"] += 1
    elif value < 80:
        bands["70-79%"] += 1
    elif value < 90:
        bands["80-89%"] += 1
    else:
        bands["90% and above"] += 1

print("\nCOCOA PERCENTAGE BANDS")

for band, count in bands.items():
    print(band, count)

plt.figure()
plt.pie(
    list(bands.values()),
    labels=list(bands.keys()),
    autopct="%1.1f%%"
)
plt.title("Distribution of Cocoa Percentage Bands")
plt.tight_layout()
plt.show()

print("Interpretation: The largest slice represents the most common cocoa percentage band.")


# 4. Histogram: Rating distribution
valid_ratings = [
    value for value in ratings if value is not None
]

plt.figure()
plt.hist(valid_ratings, bins=10, edgecolor="black")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Chocolate Ratings")
plt.tight_layout()
plt.show()

print("Interpretation: The histogram shows the rating ranges occurring most frequently.")


# 5. Scatter plot: Cocoa percentage versus rating
scatter_cocoa = []
scatter_rating = []

for i in range(len(df)):
    if cocoa[i] is not None and ratings[i] is not None:
        scatter_cocoa.append(cocoa[i])
        scatter_rating.append(ratings[i])

plt.figure()
plt.scatter(scatter_cocoa, scatter_rating)
plt.xlabel("Cocoa Percentage")
plt.ylabel("Rating")
plt.title("Cocoa Percentage versus Rating")
plt.tight_layout()
plt.show()

print("Interpretation: The scatter plot shows whether cocoa percentage is associated with rating.")


# ============================================================
# PART B: ASSOCIATION RULES
# ============================================================

TRANSACTION_FILE = "cocoa_transactions.csv"
MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.50
MAX_ITEMSET_SIZE = 3

transaction_df = pd.read_csv(TRANSACTION_FILE)

if "Items" not in transaction_df.columns:
    raise ValueError(
        "The transaction CSV must contain a column named Items."
    )

transactions = []

for value in transaction_df["Items"].fillna("").tolist():
    items = [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]

    # Each transaction is represented as a set
    transactions.append(set(items))

N = len(transactions)

if N == 0:
    raise ValueError("No transactions found.")

print("\nPART B: ASSOCIATION RULES")
print("Total transactions:", N)


# Count how many transactions contain an itemset
def count_itemset(itemset, transactions):
    count = 0

    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1

    return count


# Generate frequent itemsets manually
all_items = sorted({
    item
    for transaction in transactions
    for item in transaction
})

frequent_itemsets = {}
itemset_counts = {}

for size in range(1, MAX_ITEMSET_SIZE + 1):
    for combination in combinations(all_items, size):
        itemset = frozenset(combination)
        count = count_itemset(itemset, transactions)
        support = count / N

        if support >= MIN_SUPPORT:
            frequent_itemsets[itemset] = support
            itemset_counts[itemset] = count


# Generate rules from frequent itemsets
rules = []

for itemset, support_xy in frequent_itemsets.items():
    if len(itemset) < 2:
        continue

    items = list(itemset)

    for size in range(1, len(items)):
        for antecedent_tuple in combinations(items, size):
            antecedent = frozenset(antecedent_tuple)
            consequent = itemset - antecedent

            count_x = count_itemset(antecedent, transactions)
            count_y = count_itemset(consequent, transactions)

            if count_x == 0 or count_y == 0:
                continue

            support_x = count_x / N
            support_y = count_y / N

            confidence = support_xy / support_x
            lift = confidence / support_y

            if confidence >= MIN_CONFIDENCE:
                rules.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": support_xy,
                    "confidence": confidence,
                    "lift": lift
                })


# Sort rules by lift
rules.sort(key=lambda rule: rule["lift"], reverse=True)

print("\nRULES SATISFYING THE THRESHOLDS")

if not rules:
    print("No rules found. Check transactions or lower thresholds.")

for rule in rules:
    left = ", ".join(sorted(rule["antecedent"]))
    right = ", ".join(sorted(rule["consequent"]))

    print("\nRule:", left, "->", right)
    print("Support:", round(rule["support"], 4))
    print("Confidence:", round(rule["confidence"], 4))
    print("Lift:", round(rule["lift"], 4))


# Top 5 rules by lift
print("\nTOP 5 RULES BY LIFT")

for i, rule in enumerate(rules[:5], start=1):
    left = ", ".join(sorted(rule["antecedent"]))
    right = ", ".join(sorted(rule["consequent"]))

    print(
        i, left, "->", right,
        "| Support:", round(rule["support"], 4),
        "| Confidence:", round(rule["confidence"], 4),
        "| Lift:", round(rule["lift"], 4)
    )


# Interpret two chosen rules
print("\nINTERPRETATION OF TWO RULES")

if len(rules) >= 2:
    for rule in [rules[0], rules[1]]:
        left = ", ".join(sorted(rule["antecedent"]))
        right = ", ".join(sorted(rule["consequent"]))

        print("\nRule:", left, "->", right)
        print("Lift:", round(rule["lift"], 4))

        if rule["lift"] > 1:
            print("Positive association: items occur together more often than expected.")
        elif rule["lift"] < 1:
            print("Negative association: items occur together less often than expected.")
        else:
            print("The items are independent according to lift.")

elif len(rules) == 1:
    print("Only one rule met the thresholds; choose another if available.")
else:
    print("No rules met the thresholds, so there are no rules to interpret.")
