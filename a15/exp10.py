import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("breast_cancer_wisconsin - breast_cancer_wisconsin.csv")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# 1. 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 2. Train and compare criterion='gini' vs criterion='entropy'
for crit in ["gini", "entropy"]:
    dt = DecisionTreeClassifier(criterion=crit, max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    print(f"=== Criterion: {crit.upper()} ===")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, pos_label="M"), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, pos_label="M"), 4))
    print("F1-Score :", round(f1_score(y_test, y_pred, pos_label="M"), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=["B", "M"]))
    
    top5 = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
    print("Top 5 Features:\n", top5.round(4), "\n")
    
    plt.figure(figsize=(14, 6))
    plot_tree(dt, feature_names=list(X.columns), class_names=["B", "M"], filled=True, fontsize=8)
    plt.title(f"Decision Tree ({crit})")
    plt.show()