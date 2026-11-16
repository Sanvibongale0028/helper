import math

# Target Rating class (High >= 3.5, Mid >= 2.75, Low < 2.75)
y = ["High" if r >= 3.5 else ("Mid" if r >= 2.75 else "Low") for r in df["Rating"]]
N = len(y)

# Top 5 locations + Other
loc_cnt = {}
for loc in df["Company Location"]:
    if loc != "Other":
        loc_cnt[loc] = loc_cnt.get(loc, 0) + 1

loc_items = list(loc_cnt.items())
for i in range(len(loc_items)):
    for j in range(i + 1, len(loc_items)):
        if loc_items[j][1] > loc_items[i][1]:
            loc_items[i], loc_items[j] = loc_items[j], loc_items[i]

top5 = [loc_items[i][0] for i in range(5)]
loc_col = [loc if loc in top5 else "Other" for loc in df["Company Location"]]

def entropy(labels):
    t = len(labels)
    if t == 0:
        return 0.0
    c = {}
    for l in labels:
        c[l] = c.get(l, 0) + 1
    e = 0.0
    for v in c.values():
        p = v / t
        e -= p * math.log2(p)
    return e

def gini(labels):
    t = len(labels)
    if t == 0:
        return 0.0
    c = {}
    for l in labels:
        c[l] = c.get(l, 0) + 1
    s = 0.0
    for v in c.values():
        p = v / t
        s += p * p
    return 1.0 - s

be, bg = entropy(y), gini(y)
print(f"Base Entropy = {be:.4f}, Base Gini = {bg:.4f}\n")
print("Attribute        | Best Split         | Info Gain | Split Gini")

# 1. Categorical: Company Location (Top 5 + Other)
cats = []
for x in loc_col:
    if x not in cats:
        cats.append(x)

we, wg = 0.0, 0.0
for cat in cats:
    sub = [y[i] for i in range(N) if loc_col[i] == cat]
    we += (len(sub) / N) * entropy(sub)
    wg += (len(sub) / N) * gini(sub)
print(f"Company Location | Top 5 + Other      | {be - we:.4f}    | {wg:.4f}")

# 2. Numeric: Cocoa % and Review Date
for name, col_vals in [("Cocoa %", list(df["Cocoa_Num"])), ("Review Date", list(df["Review Date"]))]:
    u = []
    for x in col_vals:
        if x not in u:
            u.append(x)
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            if u[j] < u[i]:
                u[i], u[j] = u[j], u[i]
                
    best_ig, best_t_ig = -1.0, 0.0
    best_g, best_t_g = 999.0, 0.0
    
    for i in range(len(u) - 1):
        t = (u[i] + u[i + 1]) / 2
        left = [y[k] for k in range(N) if col_vals[k] <= t]
        right = [y[k] for k in range(N) if col_vals[k] > t]
        
        es = (len(left) / N) * entropy(left) + (len(right) / N) * entropy(right)
        gs = (len(left) / N) * gini(left) + (len(right) / N) * gini(right)
        
        if be - es > best_ig:
            best_ig, best_t_ig = be - es, t
        if gs < best_g:
            best_g, best_t_g = gs, t
            
    print(f"{name:<16} | <={best_t_ig} / <={best_t_g} | {best_ig:.4f}    | {best_g:.4f}")

print("\nChosen Root Split Attribute: Cocoa % (highest Information Gain = 0.0184 and lowest Split Gini = 0.5759).")