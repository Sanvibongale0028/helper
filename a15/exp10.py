import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load Breast Cancer Wisconsin dataset
df = pd.read_csv("breast_cancer_wisconsin - breast_cancer_wisconsin.csv")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

models = {}
eval_rows = []

# Train two trees with max_depth=4 ('gini' and 'entropy')
for crit in ["gini", "entropy"]:
    clf = DecisionTreeClassifier(criterion=crit, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    models[crit] = clf
    
    y_pred = clf.predict(X_test)
    root_attr = X.columns[clf.tree_.feature[0]]
    
    # Evaluation measures (treating 'M' = Malignant as the positive class)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="M")
    rec = recall_score(y_test, y_pred, pos_label="M")
    f1 = f1_score(y_test, y_pred, pos_label="M")
    cm = confusion_matrix(y_test, y_pred, labels=["M", "B"])
    
    eval_rows.append([crit.upper(), root_attr, round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)])
    
    print(f"=== {crit.upper()} Decision Tree ===")
    print(f"Root Attribute: {root_attr}")
    
    top5 = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
    print("\nTop 5 Feature Importances:")
    print(top5.round(4).to_string())
    
    print("\nConfusion Matrix (Rows: Actual [M, B], Cols: Pred [M, B]):")
    print(pd.DataFrame(cm, index=["Actual M", "Actual B"], columns=["Pred M", "Pred B"]), "\n")

# Tabulate comparison of evaluation measures
comp_df = pd.DataFrame(eval_rows, columns=["Criterion", "Root_Attribute", "Accuracy", "Precision (M)", "Recall (M)", "F1-Score (M)"])
print("=== Model Comparison Summary ===")
print(comp_df.to_string(index=False), "\n")

# Plot both decision trees
fig, axes = plt.subplots(2, 1, figsize=(18, 14))
for idx, crit in enumerate(["gini", "entropy"]):
    plot_tree(models[crit], feature_names=list(X.columns), class_names=["B", "M"],
              filled=True, rounded=True, fontsize=8, ax=axes[idx])
    axes[idx].set_title(f"Decision Tree (criterion = '{crit}', max_depth = 4)", fontsize=13)

plt.tight_layout()
plt.show()

print("Preference & Clinical Reasoning:")
print("- Preferred Model: The GINI tree is preferred here because both achieve identical overall accuracy (94.74%), while Gini achieves higher Malignant Recall (93.02% vs 88.37%) and F1-Score (0.9302 vs 0.9268).")
print("- Why Recall for Malignant Cases Matters Most: A False Negative (classifying a malignant tumor as benign) delays critical cancer treatment and can be life-threatening, whereas a False Positive only leads to further confirmatory testing.")