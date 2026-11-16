import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Load dataset and discretise all 5 attributes into Low / Medium / High (equal frequency)
df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
cols = ["Economy", "Fam", "Health", "Freedom", "H_Score"]

for col in cols:
    df[col + "_Band"] = pd.qcut(df[col], q=3, labels=[f"{col}=Low", f"{col}=Medium", f"{col}=High"])

transactions = df[[c + "_Band" for c in cols]].astype(str).values.tolist()
N = len(transactions)
min_sup = 0.15

# --- 1. Manual Apriori Implementation ---
def calc_support(itemset):
    count = 0
    for t in transactions:
        if all(item in t for item in itemset):
            count += 1
    return count / N

# Find L1
unique_items = sorted({item for t in transactions for item in t})
current_L = {}
for item in unique_items:
    sup = calc_support((item,))
    if sup >= min_sup:
        current_L[(item,)] = sup

all_frequent = {}
k = 1

while current_L:
    all_frequent.update(current_L)
    keys = list(current_L.keys())
    next_L = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cand = tuple(sorted(set(keys[i]) | set(keys[j])))
            if len(cand) == k + 1 and cand not in next_L:
                sup = calc_support(cand)
                if sup >= min_sup:
                    next_L[cand] = sup
    current_L = next_L
    k += 1

print(f"=== Manual Apriori Frequent Itemsets (min_support = {min_sup}, Total = {len(all_frequent)}) ===")
for iset, sup in sorted(all_frequent.items(), key=lambda x: (-len(x[0]), -x[1])):
    print(f"{list(iset)} -> Support = {sup:.4f}")

# --- 2. Verification with mlxtend ---
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

mlx_freq = apriori(df_encoded, min_support=min_sup, use_colnames=True)
print(f"\n=== mlxtend Verification ===")
print(f"Manual Apriori Count : {len(all_frequent)}")
print(f"mlxtend Apriori Count: {len(mlx_freq)}")
print("Exact Match:", len(all_frequent) == len(mlx_freq))