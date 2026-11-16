import pandas as pd

# Load Cocoa dataset (drop missing Review Date rows if any)
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv").dropna(subset=["Review Date"])

# Find top 8 Company Locations
loc_counts = {}
for loc in df["Company Location"]:
    loc_counts[loc] = loc_counts.get(loc, 0) + 1
top8 = sorted(loc_counts, key=loc_counts.get, reverse=True)[:8]

# Build transactions with 4 items per review
transactions = []
for _, row in df.iterrows():
    loc = row["Company Location"] if row["Company Location"] in top8 else "Other"
    
    cp = float(str(row["Cocoa Percent"]).replace("%", ""))
    c_band = "Cocoa_<70" if cp < 70 else ("Cocoa_=70" if cp == 70 else "Cocoa_>70")
    
    r = float(row["Rating"])
    r_class = "Rating_High" if r >= 3.5 else ("Rating_Mid" if r >= 2.75 else "Rating_Low")
    
    yr = int(row["Review Date"])
    period = "Period_2006-09" if yr <= 2009 else ("Period_2010-13" if yr <= 2013 else "Period_2014-17")
    
    transactions.append([loc, c_band, r_class, period])

N = len(transactions)
min_sup = 0.05

# Helper to calculate support manually
def get_support(itemset):
    count = 0
    for t in transactions:
        if all(item in t for item in itemset):
            count += 1
    return count / N

# Generate L1
all_items = sorted({item for t in transactions for item in t})
L1 = {}
for item in all_items:
    sup = get_support([item])
    if sup >= min_sup:
        L1[(item,)] = sup

# Generate L2 from L1
L1_keys = list(L1.keys())
L2 = {}
for i in range(len(L1_keys)):
    for j in range(i + 1, len(L1_keys)):
        cand = tuple(sorted(set(L1_keys[i]) | set(L1_keys[j])))
        sup = get_support(cand)
        if sup >= min_sup:
            L2[cand] = sup

# Generate L3 from L2
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
    print(f"=== {name} (Count: {len(L)}) ===")
    for k, v in sorted(L.items(), key=lambda x: x[1], reverse=True):
        print(f"{list(k)}: Support = {v:.4f}")
    print()

# Verify support of 3 itemsets by direct counting
print("=== Direct Verification of 3 Itemsets ===")
for itemset in list(L3.keys())[:3]:
    direct_count = sum(1 for t in transactions if all(x in t for x in itemset))
    print(f"{list(itemset)} -> Count = {direct_count}/{N} = {direct_count/N:.4f} (Apriori: {L3[itemset]:.4f})")