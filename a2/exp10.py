import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load dataset
df = pd.read_csv("iris - iris.csv")
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[features]
y = df["species"]

# 70/30 train-test split with random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Train Decision Tree with max_depth=3
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 1. Print tree as indented text
print("Decision Tree Structure:")
print(export_text(clf, feature_names=features))

# 2. Predictions & Evaluation Metrics
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
metrics_df = pd.DataFrame({
    "Class": clf.classes_,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1
})
print("Per-Class Metrics:\n", metrics_df.to_string(index=False), "\n")

print("3x3 Confusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=clf.classes_, columns=clf.classes_), "\n")

# 3. Root attribute explanation
root_feature = features[clf.tree_.feature[0]]
root_thresh = clf.tree_.threshold[0]
print(f"Root Attribute: {root_feature} (threshold <= {root_thresh:.2f})")
print("Why chosen: It yields the highest Information Gain (greatest reduction in Gini impurity) across all features, cleanly separating all 'setosa' samples from the other two species in a single split.")