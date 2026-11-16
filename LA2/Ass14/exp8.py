
import pandas as pd
from itertools import combinations

df = pd.read_csv("happiness.csv")
df.columns = df.columns.str.strip()

def find_column(possible_names):
    for name in possible_names:
        for col in df.columns:
            if col.strip().lower() == name.lower():
                return col

    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.strip().lower():
                return col

    raise ValueError("Column not found: " + str(possible_names))

economy_col = find_column(["Economy", "Economy (GDP per Capita)", "GDP per capita"])
family_col = find_column(["Fam", "Family", "Social support"])
health_col = find_column(["Health", "Health (Life Expectancy)", "Healthy life expectancy"])
freedom_col = find_column(["Freedom", "Freedom to make life choices"])
score_col = find_column(["N Score", "H Score", "Happiness Score", "Score"])

columns = {
    "Economy": economy_col,
    "Fam": family_col,
    "Health": health_col,
    "Freedom": freedom_col,
    "N Score": score_col
}

# Convert columns to numeric values
for col in columns.values():
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=list(columns.values())).reset_index(drop=True)

# Equal-frequency discretization into Low, Medium and High
def discretize(values):
    indexed_values = sorted(
        enumerate(values),
        key=lambda x: x[1]
    )

    labels = [""] * len(values)
    n = len(values)

    for rank, (original_index, value) in enumerate(indexed_values):
        if rank < n // 3:
            labels[original_index] = "Low"
        elif rank < (2 * n) // 3:
            labels[original_index] = "Medium"
        else:
            labels[original_index] = "High"

    return labels

discretized = {}

for attribute, col in columns.items():
    discretized[attribute] = discretize(df[col].tolist())

# Each country is one transaction
transactions = []

for i in range(len(df)):
    transaction = set()

    for attribute in columns:
        label = discretized[attribute][i]
        transaction.add(attribute + " " + label)

    transactions.append(transaction)

total_transactions = len(transactions)
min_support = 0.1
min_confidence = 0.6

def support(itemset):
    count = sum(
        1 for transaction in transactions
        if itemset.issubset(transaction)
    )
    return count / total_transactions

# Apriori algorithm
all_items = sorted(set(
    item for transaction in transactions
    for item in transaction
))

frequent_itemsets = {}
current_itemsets = [{item} for item in all_items]

k = 1

while current_itemsets:
    frequent_current = []

    for itemset in current_itemsets:
        sup = support(itemset)

        if sup >= min_support:
            frequent_itemsets[frozenset(itemset)] = sup
            frequent_current.append(frozenset(itemset))

    if not frequent_current:
        break

    next_candidates = set()

    for a, b in combinations(frequent_current, 2):
        union = a | b

        if len(union) == k + 1:
            # Apriori pruning: all subsets must be frequent
            subsets_frequent = all(
                frozenset(subset) in frequent_itemsets
                for subset in combinations(union, k)
            )

            if subsets_frequent:
                next_candidates.add(union)

    current_itemsets = [set(x) for x in next_candidates]
    k += 1

print("Total transactions:", total_transactions)
print("\nFrequent itemsets:")
for itemset, sup in sorted(
    frequent_itemsets.items(),
    key=lambda x: (len(x[0]), sorted(x[0]))
):
    print(sorted(itemset), "Support =", round(sup, 3))

# Generate rules with N Score High as the consequent
target = "N Score High"
rules = []

for itemset, itemset_support in frequent_itemsets.items():
    if target not in itemset or len(itemset) < 2:
        continue

    remaining = set(itemset) - {target}

    # Consider every non-empty antecedent subset
    for r in range(1, len(remaining) + 1):
        for antecedent_tuple in combinations(remaining, r):
            antecedent = set(antecedent_tuple)

            if not antecedent:
                continue

            antecedent_support = support(antecedent)

            if antecedent_support == 0:
                continue

            confidence = itemset_support / antecedent_support
            target_support = support({target})
            lift = confidence / target_support if target_support else 0

            if confidence >= min_confidence:
                rules.append({
                    "Antecedent": " AND ".join(sorted(antecedent)),
                    "Consequent": target,
                    "Support": itemset_support,
                    "Confidence": confidence,
                    "Lift": lift
                })

# Rank by lift, highest first
rules.sort(key=lambda x: x["Lift"], reverse=True)

print("\nRules with consequent N Score High")
print("Minimum support:", min_support)
print("Minimum confidence:", min_confidence)

if not rules:
    print("No rules satisfy both thresholds.")
else:
    for rank, rule in enumerate(rules, start=1):
        print("\nRank:", rank)
        print(rule["Antecedent"], "->", rule["Consequent"])
        print("Support:", round(rule["Support"], 3))
        print("Confidence:", round(rule["Confidence"], 3))
        print("Lift:", round(rule["Lift"], 3))

    print("\nTOP 3 RULES")
    for i, rule in enumerate(rules[:3], start=1):
        print(
            i, ".",
            rule["Antecedent"], "->", rule["Consequent"],
            "| Support:", round(rule["Support"], 3),
            "| Confidence:", round(rule["Confidence"], 3),
            "| Lift:", round(rule["Lift"], 3)
        )
