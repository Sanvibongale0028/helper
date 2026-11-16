import pandas as pd
from itertools import combinations

df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv").dropna(subset=["Review Date"])
df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").astype(float)
top8 = df["Company Location"].value_counts().head(8).index.tolist()

# Build transactions from Assignment 3 Part B
txns = []
for _, row in df.iterrows():
    loc = row["Company Location"] if row["Company Location"] in top8 else "Other"
    cp = row["Cocoa_Num"]
    cb = "Cocoa_<70" if cp < 70 else ("Cocoa_=70" if cp == 70 else "Cocoa_>70")
    r = row["Rating"]
    rc = "Rating_High" if r >= 3.5 else ("Rating_Mid" if r >= 2.75 else "Rating_Low")
    yr = int(row["Review Date"])
    pb = "Period_2006-09" if yr <= 2009 else ("Period_2010-13" if yr <= 2013 else "Period_2014-17")
    txns.append(frozenset([loc, cb, rc, pb]))

N = len(txns)
items = sorted({i for t in txns for i in t})

# Frequent itemsets (min_support = 0.03)
freq = {}
for k in range(1, 5):
    for comb in combinations(items, k):
        s = frozenset(comb)
        sup = sum(1 for t in txns if s.issubset(t)) / N
        if sup >= 0.03:
            freq[s] = sup

# Generate rules (min_confidence = 0.5)
rules = []
for iset, sup_xy in freq.items():
    if len(iset) > 1:
        for r in range(1, len(iset)):
            for ant in combinations(iset, r):
                ant = frozenset(ant)
                cons = iset - ant
                conf = sup_xy / freq[ant]
                if conf >= 0.5:
                    lift = conf / freq[cons]
                    rules.append((list(ant), list(cons), round(sup_xy, 4), round(conf, 4), round(lift, 4)))

rules.sort(key=lambda x: x[4], reverse=True)

print("All Rules (Antecedent -> Consequent | Support, Confidence, Lift):")
for r in rules:
    print(f"{r[0]} -> {r[1]} | sup={r[2]}, conf={r[3]}, lift={r[4]}")

print("\nTop 5 Rules by Lift:")
for r in rules[:5]:
    print(f"{r[0]} -> {r[1]} | sup={r[2]}, conf={r[3]}, lift={r[4]}")

print("\nExplanation:")
print("1. Lift > 1 (e.g., ['Rating_High', 'Cocoa_=70', 'U.S.A.'] -> ['Period_2014-17'], lift=1.496): Positive association; they appear together ~1.5x more often than expected by chance.")
print("2. Lift < 1 (e.g., ['Period_2010-13'] -> ['Rating_Mid'], lift=0.9985): Negative association; the antecedent slightly decreases the chance of the consequent occurring.")