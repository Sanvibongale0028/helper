import math
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
df = pd.read_csv("weather_numeric - weather_numeric.csv")
classes = ["yes", "no"]
cat_cols = ["outlook", "windy"]
num_cols = ["temperature", "humidity"]

# Helper function to fit Mixed Naive Bayes and predict a single instance
def naive_bayes_predict(train_data, test_instance, show_working=False):
    n_total = len(train_data)
    priors = {}
    cat_probs = {c: {} for c in classes}
    num_stats = {c: {} for c in classes}
    
    # Fit model parameters
    for c in classes:
        subset = train_data[train_data["play"] == c]
        n_c = len(subset)
        priors[c] = n_c / n_total
        
        # Categorical probabilities with Laplace smoothing (alpha = 1)
        for col in cat_cols:
            unique_vals = sorted(df[col].unique())
            k = len(unique_vals)
            cat_probs[c][col] = {}
            for val in unique_vals:
                count = (subset[col] == val).sum()
                cat_probs[c][col][val] = (count + 1) / (n_c + k)
                
        # Gaussian parameters (mean & sample std) for numeric features
        for col in num_cols:
            num_stats[c][col] = (subset[col].mean(), subset[col].std(ddof=1))
            
    if show_working:
        print("=== Prior Probabilities ===")
        for c in classes:
            print(f"P(play = {c}) = {priors[c]:.4f}")
            
        print("\n=== Categorical Probability Tables (with Laplace Smoothing) ===")
        for col in cat_cols:
            print(f"\nAttribute: {col}")
            table = pd.DataFrame({c: cat_probs[c][col] for c in classes})
            print(table)
            
        print("\n=== Gaussian Parameters (Mean, Std) ===")
        for col in num_cols:
            print(f"{col}: yes -> (mean={num_stats['yes'][col][0]:.2f}, std={num_stats['yes'][col][1]:.2f}) | "
                  f"no -> (mean={num_stats['no'][col][0]:.2f}, std={num_stats['no'][col][1]:.2f})")
            
    # Compute posterior likelihoods for test_instance
    posteriors = {}
    if show_working:
        print("\n=== Step-by-Step Classification Working for New Day ===")
        
    for c in classes:
        prob = priors[c]
        steps = [f"P({c})={priors[c]:.4f}"]
        
        for col in cat_cols:
            p_cat = cat_probs[c][col][test_instance[col]]
            prob *= p_cat
            steps.append(f"P({test_instance[col]}|{c})={p_cat:.4f}")
            
        for col in num_cols:
            mean, std = num_stats[c][col]
            x = test_instance[col]
            p_num = (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((x - mean) ** 2) / (2 * std ** 2))
            prob *= p_num
            steps.append(f"P({col}={x}|{c})={p_num:.5f}")
            
        posteriors[c] = prob
        if show_working:
            print(f"Class '{c}': " + " * ".join(steps) + f" = {prob:.8f}")
            
    total_evidence = posteriors["yes"] + posteriors["no"]
    if show_working:
        print(f"\nNormalized P(yes | X) = {posteriors['yes'] / total_evidence:.4f}")
        print(f"Normalized P(no  | X) = {posteriors['no'] / total_evidence:.4f}")
        
    return "yes" if posteriors["yes"] >= posteriors["no"] else "no"

# 1. Print tables and classify the new day with full working
new_day = {"outlook": "sunny", "temperature": 66, "humidity": 90, "windy": True}
prediction = naive_bayes_predict(df, new_day, show_working=True)
print(f"\n>>> Predicted Class for New Day: {prediction.upper()} <<<\n")

# 2. Evaluate using Leave-One-Out Cross-Validation (14 folds)
y_true = list(df["play"])
y_pred = []

for i in range(len(df)):
    train_fold = df.drop(index=i)
    test_fold = df.iloc[i]
    y_pred.append(naive_bayes_predict(train_fold, test_fold, show_working=False))

print("=== Leave-One-Out Cross-Validation (14 Folds) Results ===")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, pos_label='yes'):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, pos_label='yes'):.4f}")
print(f"F1-Score : {f1_score(y_true, y_pred, pos_label='yes'):.4f}\n")

cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=["yes", "no"]),
                  index=["Actual Yes", "Actual No"],
                  columns=["Pred Yes", "Pred No"])
print("Confusion Matrix:\n", cm)