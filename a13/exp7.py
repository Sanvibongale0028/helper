import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

df = pd.read_csv("GHI_Report (1) - GHI_Report (1).csv")
cols = ["Economy", "Fam", "Health", "Freedom", "H_Score"]

# 1. Discretise each attribute into Low / Medium / High (equal frequency)
for col in cols:
    df[col + "_b"] = pd.qcut(df[col], q=3, labels=[f"{col}=Low", f"{col}=Medium", f"{col}=High"])

txns = df[[c + "_b" for c in cols]].astype(str).values.tolist()
N = len(txns)
min_sup = 0.15

# 2. Manual Apriori
def get_support(iset):
    cnt = 0
    for t in txns:
        ok = True
        for item in iset:
            if item not in t:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt / N

items = []
for t in txns:
    for it in t:
        if it not in items:
            items.append(it)

curr_L = {}
for it in items:
    s = get_support([it])
    if s >= min_sup:
        curr_L[(it,)] = s

freq_all = {}
k = 1
while curr_L:
    freq_all.update(curr_L)
    keys = list(curr_L.keys())
    next_L = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cand = tuple(sorted(set(keys[i]) | set(keys[j])))
            if len(cand) == k + 1 and cand not in next_L:
                s = get_support(cand)
                if s >= min_sup:
                    next_L[cand] = s
    curr_L = next_L
    k += 1

print(f"=== Manual Frequent Itemsets (min_support = {min_sup}, Total = {len(freq_all)}) ===")
for iset, sup in freq_all.items():
    print(f"{list(iset)}: {sup:.4f}")

# 3. Verify with mlxtend
te = TransactionEncoder()
df_enc = pd.DataFrame(te.fit(txns).transform(txns), columns=te.columns_)
mlx_res = apriori(df_enc, min_support=min_sup, use_colnames=True)

print(f"\nVerification -> Manual Count: {len(freq_all)}, mlxtend Count: {len(mlx_res)}, Match: {len(freq_all) == len(mlx_res)}")