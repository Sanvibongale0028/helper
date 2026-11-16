
# pip install mlxtend

import pandas as pd
from itertools import combinations

# Load GHI / World Happiness dataset
df = pd.read_csv("happiness.csv")
df.columns = [str(c).strip() for c in df.columns]


def find_column(candidates):
    for candidate in candidates:
        for col in df.columns:
            if col.lower().strip() == candidate.lower().strip():
                return col
    raise ValueError(
        "Column not found: " + str(candidates)
        + "\nAvailable columns: " + str(list(df.columns))
    )


economy_col = find_column([
    "Economy", "Economy (GDP per Capita)",
    "Logged GDP per capita", "GDP per capita"
])

family_col = find_column(["Family", "Social support"])

health_col = find_column([
    "Health", "Health (Life Expectancy)",
    "Healthy life expectancy"
])

freedom_col = find_column([
    "Freedom", "Freedom to make life choices"
])

score_col = find_column([
    "H Score", "Happiness Score", "Score",
    "Happiness.Score", "Ladder score"
])

attributes = {
    "Economy": economy_col,
    "Family": family_col,
    "Health": health_col,
    "Freedom": freedom_col,
    "HScore": score_col
}


# Prepare numerical data
clean_data = {}

for name, column in attributes.items():
    clean_data[name] = []

    for value in df[column].tolist():
        try:
            number = float(value)

            if number == number and abs(number) != float("inf"):
                clean_data[name].append(number)
            else:
                clean_data[name].append(None)

        except (ValueError, TypeError):
            clean_data[name].append(None)


# Keep only countries with all five attributes present
valid_rows = []

for i in range(len(df)):
    row = {}

    valid = True

    for name in attributes:
        value = clean_data[name][i]

        if value is None:
            valid = False
            break

        row[name] = value

    if valid:
        valid_rows.append(row)

print("Valid countries:", len(valid_rows))


# Equal-frequency discretization
# First third = Low, second third = Medium, final third = High
def discretize(values):
    ordered = sorted(values)
    n = len(ordered)
    result = []

    for value in values:
        rank = 0

        for item in ordered:
            if item < value:
                rank += 1

        fraction = rank / n

        if fraction < 1 / 3:
            result.append("Low")
        elif fraction < 2 / 3:
            result.append("Medium")
        else:
            result.append("High")

    return result


discretized = {}

for name in attributes:
    values = [row[name] for row in valid_rows]
    discretized[name] = discretize(values)


# Each country is a transaction of five items
transactions = []

for i in range(len(valid_rows)):
    transaction = []

    for name in attributes:
        transaction.append(
            name + "-" + discretized[name][i]
        )

    transactions.append(transaction)

print("\nEXAMPLE TRANSACTIONS")

for transaction in transactions[:5]:
    print(transaction)


# Manual Apriori algorithm
MIN_SUPPORT = 0.15
N = len(transactions)

if N == 0:
    raise ValueError("No valid transactions available.")


# Count support of an itemset
def support_count(itemset):
    count = 0

    for transaction in transactions:
        if itemset.issubset(set(transaction)):
            count += 1

    return count


def support(itemset):
    return support_count(itemset) / N


# Find frequent one-item sets
all_items = sorted({
    item for transaction in transactions for item in transaction
})

frequent_itemsets = {}
current_level = []

for item in all_items:
    itemset = frozenset([item])
    item_support = support(itemset)

    if item_support >= MIN_SUPPORT:
        frequent_itemsets[itemset] = item_support
        current_level.append(itemset)


# Generate larger itemsets
k = 2

while current_level:
    candidates = set()

    for a in range(len(current_level)):
        for b in range(a + 1, len(current_level)):
            candidate = current_level[a] | current_level[b]

            if len(candidate) != k:
                continue

            # Apriori property: every subset must be frequent
            all_subsets_frequent = True

            for subset in combinations(sorted(candidate), k - 1):
                if frozenset(subset) not in frequent_itemsets:
                    all_subsets_frequent = False
                    break

            if all_subsets_frequent:
                candidates.add(candidate)

    next_level = []

    for candidate in candidates:
        item_support = support(candidate)

        if item_support >= MIN_SUPPORT:
            frequent_itemsets[candidate] = item_support
            next_level.append(candidate)

    current_level = next_level
    k += 1


# Print all frequent itemsets
print("\nFREQUENT ITEMSETS (MINIMUM SUPPORT = 0.15)")

for itemset in sorted(
    frequent_itemsets,
    key=lambda item: (len(item), sorted(item))
):
    print(
        sorted(itemset),
        "Support =", round(frequent_itemsets[itemset], 4)
    )


# Verify using mlxtend
print("\nVERIFICATION USING MLXTEND")

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)

    onehot_df = pd.DataFrame(
        encoded, columns=encoder.columns_
    )

    result = apriori(
        onehot_df,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )

    manual_results = {
        itemset for itemset in frequent_itemsets
    }

    library_results = {
        frozenset(itemset)
        for itemset in result["itemsets"]
    }

    print("Manual frequent itemsets:", len(manual_results))
    print("mlxtend frequent itemsets:", len(library_results))
    print("Results match:", manual_results == library_results)

    if manual_results != library_results:
        print("Only manual:", manual_results - library_results)
        print("Only mlxtend:", library_results - manual_results)

except ImportError:
    print("Install mlxtend to verify:")
    print("pip install mlxtend")
