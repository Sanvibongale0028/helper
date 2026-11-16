import math
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("weather_numeric - weather_numeric.csv")

# Helper to compute Gaussian probability density
def gaussian(x, mean, std):
    return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((x - mean) ** 2) / (2 * (std ** 2)))

# Function to train Mixed Naive Bayes and predict one row
def nb_predict(train, row):
    priors = {}
    cat_probs = {"outlook": {}, "windy": {}}
    gauss = {"temperature": {}, "humidity": {}}
    
    for c in ["yes", "no"]:
        sub = train[train["play"] == c]
        priors[c] = len(sub) / len(train)
        
        # Laplace smoothing for categorical attributes
        for val in ["sunny", "overcast", "rainy"]:
            cat_probs["outlook"][(c, val)] = ((sub["outlook"] == val).sum() + 1) / (len(sub) + 3)
        for val in [True, False]:
            cat_probs["windy"][(c, val)] = ((sub["windy"] == val).sum() + 1) / (len(sub) + 2)
            
        # Mean and sample std for numeric attributes
        for col in ["temperature", "humidity"]:
            gauss[col][c] = (sub[col].mean(), sub[col].std(ddof=1))
            
    # Compute posterior likelihoods for the test row
    post = {}
    for c in ["yes", "no"]:
        p = priors[c]
        p *= cat_probs["outlook"][(c, row["outlook"])]
        p *= cat_probs["windy"][(c, row["windy"])]
        p *= gaussian(row["temperature"], *gauss["temperature"][c])
        p *= gaussian(row["humidity"], *gauss["humidity"][c])
        post[c] = p
        
    pred = "yes" if post["yes"] >= post["no"] else "no"
    return priors, cat_probs, gauss, post, pred

# 1. Train on full dataset & print probability tables
query = {"outlook": "sunny", "temperature": 66, "humidity": 90, "windy": True}
priors, cat_probs, gauss, post, pred = nb_predict(df, query)

print("=== Prior Probabilities ===")
print("P(play=yes) =", round(priors["yes"], 4), "| P(play=no) =", round(priors["no"], 4))

print("\n=== Categorical Likelihoods (with Laplace Smoothing) ===")
for k, v in cat_probs["outlook"].items():
    print(f"P(outlook={k[1]} | play={k[0]}) = {v:.4f}")
for k, v in cat_probs["windy"].items():
    print(f"P(windy={k[1]} | play={k[0]}) = {v:.4f}")

print("\n=== Gaussian Parameters (Mean, Std) ===")
for col in ["temperature", "humidity"]:
    print(f"{col} | yes: mean={gauss[col]['yes'][0]:.2f}, std={gauss[col]['yes'][1]:.2f} | no: mean={gauss[col]['no'][0]:.2f}, std={gauss[col]['no'][1]:.2f}")

# 2. Classify the new day
print("\n=== New Day Classification ===")
print("Posterior(yes) =", f"{post['yes']:.6e}")
print("Posterior(no)  =", f"{post['no']:.6e}")
print("Predicted Class:", pred.upper())

# 3. Leave-One-Out Cross-Validation (14 folds)
y_true, y_pred = [], []
for i in range(len(df)):
    train_df = df.drop(index=i)
    test_row = df.iloc[i].to_dict()
    _, _, _, _, p = nb_predict(train_df, test_row)
    y_true.append(test_row["play"])
    y_pred.append(p)

print("\n=== Leave-One-Out Cross-Validation (14 Folds) ===")
print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(precision_score(y_true, y_pred, pos_label="yes"), 4))
print("Recall   :", round(recall_score(y_true, y_pred, pos_label="yes"), 4))
print("F1-Score :", round(f1_score(y_true, y_pred, pos_label="yes"), 4))
print("Confusion Matrix (Rows: Actual [yes, no], Cols: Pred [yes, no]):\n", confusion_matrix(y_true, y_pred, labels=["yes", "no"]))