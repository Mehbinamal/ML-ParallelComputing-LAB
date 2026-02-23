import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier

import warnings
warnings.filterwarnings("ignore")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

results = []

for name, Classifier in all_estimators(type_filter='classifier'):
    try:
        model = Classifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1": f1_score(y_test, y_pred, average='macro')
        })
    except:
        continue

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)
results_df.head()

top4_names = results_df.head(4)["Model"].values
print(top4_names)

top4_models = []

for name, Classifier in all_estimators(type_filter='classifier'):
    if name in top4_names:
        model = Classifier()
        model.fit(X_train, y_train)
        top4_models.append((name, model))

best_model = top4_models[0][1]
print(best_model)

bagging = BaggingClassifier(estimator=best_model, n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)

y_pred_bag = bagging.predict(X_test)

print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bag))

boosting = AdaBoostClassifier(n_estimators=50, random_state=42)
boosting.fit(X_train, y_train)

y_pred_boost = boosting.predict(X_test)

print("Boosting Accuracy:", accuracy_score(y_test, y_pred_boost))

stacking = StackingClassifier(
    estimators=top4_models,
    final_estimator=top4_models[0][1]
)

stacking.fit(X_train, y_train)

y_pred_stack = stacking.predict(X_test)

print("Stacking Accuracy:", accuracy_score(y_test, y_pred_stack))