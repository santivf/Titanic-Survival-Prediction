import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df2 = pd.read_csv('titanic_processed.csv')

X2 = df2.drop('Survived', axis=1)
y2 = df2['Survived']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.20, random_state=1)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X2_train, y2_train)

y2_pred_log = log_model.predict(X2_test)
acc_log = accuracy_score(y2_test, y2_pred_log)

print(f"Precision Logistic: {acc_log:.2f}")
ConfusionMatrixDisplay.from_predictions(y2_test, y2_pred_log)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
tree_model.fit(X2_train, y2_train)

y2_pred_tree = tree_model.predict(X2_test)
acc_tree = accuracy_score(y2_test, y2_pred_tree)

print(f"Precision Tree: {acc_tree:.2f}")

plt.figure(figsize=(20,10))
plot_tree(tree_model, feature_names=X2.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

ConfusionMatrixDisplay.from_predictions(y2_test, y2_pred_tree)
plt.title("Confusion Matrix - Decision Tree")
plt.show()