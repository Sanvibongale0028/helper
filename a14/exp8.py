import pandas as pd
from itertools import combinations
from sklearn.cluster import KMeans

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
features = ["Economy", "Fam", "Health", "Freedom"]

# Step 1: Recreate clusters from K-Means (Exp 13)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = ["Cluster_" + str(c) for c in kmeans.fit_predict(df[features])]

# Step 2: Discretise factors and H_Score into Low / Medium / High tertiles
for col in features + ["H_Score"]:
    df[col + "_Item"] = pd.qcut(df[col], q=3, labels=[f"{col}=Low", f"{col}=Medium", f"{col}=High"])

item_cols = ["Cluster"] + [col + "_Item" for col in features + ["H_Score"]]
transactions = df[item_cols].astype(str).values.tolist()
N = len(transactions)

# Step 3: Find frequent itemsets (min_support = 0.1)
min_sup = 0.1
min_conf = 0.6
all_items = sorted({item for t in transactions for item in t})

freq_itemsets = {}
for k in range(1, 5):
    for comb in combinations(all_items, k):
        sup = sum(1 for t in transactions if set(comb).issubset(t)) / N
        if sup >= min_sup:
            freq_itemsets[frozenset(comb)] = sup

# Step 4: Generate rules with consequent {H_Score=High}
target = frozenset(["H_Score=High"])
sup_target = freq_itemsets[target]
rules = []

for itemset, sup_xy in freq_itemsets.items():
    if target.issubset(itemset) and len(itemset) > 1:
        antecedent = itemset - target
        sup_x = freq_itemsets[antecedent]
        conf = sup_xy / sup_x
        lift = conf / sup_target
        if conf >= min_conf:
            rules.append({
                "Antecedent": ", ".join(sorted(antecedent)),
                "Consequent": "H_Score=High",
                "Support": round(sup_xy, 4),
                "Confidence": round(conf, 4),
                "Lift": round(lift, 4)
            })

rules_df = pd.DataFrame(rules).sort_values(by="Lift", ascending=False)
print("=== Rules with Consequent {H_Score=High} Ranked by Lift ===")
print(rules_df.to_string(index=False))

print("\n=== Top 3 Rules Interpretation ===")
for idx, row in rules_df.head(3).iterrows():
    print(f"- If [{row['Antecedent']}], then H_Score is High (Confidence: {row['Confidence']*100:.1f}%, Lift: {row['Lift']}x more likely than random).")