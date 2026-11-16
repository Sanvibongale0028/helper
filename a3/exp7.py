import pandas as pd

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv").dropna(subset=["Review Date"])

# 1. Find top 8 Company Locations manually
loc_counts = {}
for loc in df["Company Location"]:
    loc_counts[loc] = loc_counts.get(loc, 0) + 1

loc_items = list(loc_counts.items())
for i in range(len(loc_items)):
    for j in range(i + 1, len(loc_items)):
        if loc_items[j][1] > loc_items[i][1]:
            loc_items[i], loc_items[j] = loc_items[j], loc_items[i]
top8 = [loc_items[i][0] for i in range(8)]

# 2. Build transactions with 4 items per review
txns = []
for _, row in df.iterrows():
    loc = row["Company Location"] if row["Company Location"] in top8 else "Other"
    cp = float(str(row["Cocoa Percent"]).replace("%", ""))
    c_band = "Cocoa_<70" if cp < 70 else ("Cocoa_=70" if cp == 70 else "Cocoa_>70")
    r = float(row["Rating"])
    r_class = "Rating_High" if r >= 3.5 else ("Rating_Mid" if r >= 2.75 else "Rating_Low")
    yr = int(row["Review Date"])
    period = "Period_2006-09" if yr <= 2009 else ("Period_2010-13" if yr <= 2013 else "Period_2014-17")
    txns.append([loc, c_band, r_class, period])

N = len(txns)
min_sup = 0.05

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

# 3. Generate L1
items = []
for t in txns:
    for it in t:
        if it not in items:
            items.append(it)

L1 = {}
for it in items:
    sup = get_support([it])
    if sup >= min_sup:
        L1[(it,)] = sup

# 4. Generate L2 from L1
L1_keys = list(L1.keys())
L2 = {}
for i in range(len(L1_keys)):
    for j in range(i + 1, len(L1_keys)):
        cand = tuple(sorted(set(L1_keys[i]) | set(L1_keys[j])))
        sup = get_support(cand)
        if sup >= min_sup:
            L2[cand] = sup

# 5. Generate L3 from L2
L2_keys = list(L2.keys())
L3 = {}
for i in range(len(L2_keys)):
    for j in range(i + 1, len(L2_keys)):
        cand = tuple(sorted(set(L2_keys[i]) | set(L2_keys[j])))
        if len(cand) == 3 and cand not in L3:
            sup = get_support(cand)
            if sup >= min_sup:
                L3[cand] = sup

for name, L in [("L1", L1), ("L2", L2), ("L3", L3)]:
    print(f"=== {name} ===")
    for k, v in L.items():
        print(f"{list(k)}: {v:.4f}")
    print()

# 6. Direct verification of 3 itemsets
print("=== Direct Verification of 3 Itemsets ===")
for iset in list(L3.keys())[:3]:
    cnt = 0
    for t in txns:
        if all(x in t for x in iset):
            cnt += 1
    print(f"{list(iset)} -> Count = {cnt}/{N} = {cnt/N:.4f}")