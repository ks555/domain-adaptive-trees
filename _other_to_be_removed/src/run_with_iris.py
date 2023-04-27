from src.utils import split_data, print_scores
import pandas as pd
import numpy as np
from pprint import pprint
from src.decision_tree_classifier.decision_tree_classifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import sys

# iris as a data frame
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X['y'] = iris.target
y = X['y']
X = X[iris.feature_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)

# my "target domain"
target_df = X_test.copy()
target_df['y'] = y_test.copy()
target_df.reset_index(inplace=True, drop=True)

# current_path example: [('petal length (cm)', 3.0, 'right'), ('petal width (cm)', 1.8, 'left')]

# standard dt
org_clf = DecisionTreeClassifier(max_depth=5)
org_clf.fit(X_train, y_train)
# pprint(org_clf.tree)
org_pred = org_clf.predict(X_test)
org_accuracy = print_scores(y_test, org_pred['prediction'])
print('org_clf accuracy:', org_accuracy)

# domain adapted dt (v1)
da1_clf = DecisionTreeClassifier(max_depth=5)
da1_clf.fit(X_train, y_train, alpha=0.75, X_td=X_test)
# pprint(da1_clf.tree)
da1_pred = da1_clf.predict(X_test)
da1_accuracy = print_scores(y_test, da1_pred['prediction'])
print(f'DA 1 accuracy: {da1_accuracy}')
# sys.exit(0)

# domain adapted dt (v2)
da2_clf = DecisionTreeClassifier(max_depth=5)
da2_clf.fit(X_train, y_train, alpha=1.0, X_td=X_test, y_td=y_test)
# pprint(da2_clf.tree)
da2_pred = da2_clf.predict(X_test)
da2_accuracy = print_scores(y_test, da2_pred['prediction'])
print(f'DA 2 accuracy {da2_accuracy}')

# domain adapted dt (v3)
da3_clf = DecisionTreeClassifier(max_depth=5)
da3_clf.fit(X_train, y_train, alpha=0.0, X_td=X_test, y_td=y_test)
# pprint(da3_clf.tree)
da3_pred = da3_clf.predict(X_test)
da3_accuracy = print_scores(y_test, da3_pred['prediction'])

print(f'Original accuracy: {org_accuracy}')
print(f'DA 1 accuracy: {da1_accuracy}')
print(f'DA 2 accuracy {da2_accuracy}')  # with alpha=1.0, it's the same as running the standard DT
print(f'DA 3 accuracy {da3_accuracy}')
