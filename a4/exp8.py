import pandas as pd
from itertools import combinations

# Load dataset and build transactions from Assignment 3 Part B
df = pd.read_csv("flavors_of_cocoa (1) (2) - flavors_of_cocoa (1) (2).csv").dropna(subset=["Review Date"])
df["Cocoa_Num"] = df["Cocoa Percent"].str.replace("%", "").astype(float)
top8 = df["Company Location"].value_counts().head(8).index.tolist()

transactions = []
for _, row in df.iterrows():
    loc = row["Company Location"] if row["Company Location"] in top8 else "Other"
    
    cp = row["Cocoa_Num"]
    c_band = "Cocoa_<70" if cp < 70 else ("Cocoa_=70" if cp == 70 else "Cocoa_>70")
    
    r = row["Rating"]
    r_class = "Rating_High" if r >= 3.5 else ("Rating_Mid" if r >= 2.75 else "Rating_Low")
    
    yr = int(row["Review Date"])
    period = "Period_2006-09" if yr <= 2009 else ("Period_2010-13" if yr <= 2013 else "Period_2014-17")
    
    transactions.append(frozenset([loc, c_band, r_class, period]))

N = len(transactions)
min_sup = 0.03
min_conf = 0.50

# Find all frequent itemsets with support >= 0.03
all_items = sorted({item for t in transactions for item in t})
freq_itemsets = {}

for k in range(1, 5):
    for comb in combinations(all_items, k):
        iset = frozenset(comb)
        sup = sum(1 for t in transactions if iset.issubset(t)) / N
        if sup >= min_sup:
            freq_itemsets[iset] = sup

# Generate rules with confidence >= 0.50
rules = []
for iset, sup_xy in freq_itemsets.items():
    if len(iset) > 1:
        for r_len in range(1, len(iset)):
            for ant_tuple in combinations(iset, r_len):
                ant = frozenset(ant_tuple)
                cons = iset - ant
                conf = sup_xy / freq_itemsets[ant]
                if conf >= min_conf:
                    lift = conf / freq_itemsets[cons]
                    rules.append({
                        "Antecedent": ", ".join(sorted(ant)),
                        "Consequent": ", ".join(sorted(cons)),
                        "Support": round(sup_xy, 4),
                        "Confidence": round(conf, 4),
                        "Lift": round(lift, 4)
                    })

rules_df = pd.DataFrame(rules).sort_values(by="Lift", ascending=False)
print(f"=== All Generated Rules (Total: {len(rules_df)}) ===")
print(rules_df.to_string(index=False), "\n")

print("=== Top 5 Rules by Lift ===")
print(rules_df.head(5).to_string(index=False), "\n")

# Show a rule with lift > 1 and a rule with lift < 1 for explanation
lift_gt1 = rules_df[rules_df["Lift"] > 1].iloc[0]
lift_lt1 = rules_df[rules_df["Lift"] < 1].iloc[0]

print("=== Lift Interpretation for Two Chosen Rules ===")
print(f"1. Lift > 1 Rule: [{lift_gt1['Antecedent']}] -> [{lift_gt1['Consequent']}] (Lift = {lift_gt1['Lift']})")
print("   Meaning: Positive association — these items occur together ~1.50 times more often than expected if they were statistically independent.\n")

print(f"2. Lift < 1 Rule: [{lift_lt1['Antecedent']}] -> [{lift_lt1['Consequent']}] (Lift = {lift_lt1['Lift']})")
print("   Meaning: Negative association — the presence of the antecedent slightly reduces the likelihood of the consequent occurring compared to its baseline probability.")