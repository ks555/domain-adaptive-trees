import pandas as pd
import numpy as np
from pprint import pprint
from src.decision_tree_classifier.decision_tree_classifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris = load_iris()

# iris as a data frame
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X['y'] = iris.target
y = X['y']
X = X[iris.feature_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# my "target domain"
target_df = X_test.copy()
target_df['y'] = y_test.copy()
target_df.reset_index(inplace=True, drop=True)

clf = DecisionTreeClassifier(max_depth=7, )
# clf.fit(X_train, y_train, cat_atts=['test', 'me'])
clf.fit(X_train, y_train, cat_atts=['test', 'me'], alpha=0.7, X_target_domain=X_test, y_target_domain=y_test)
pprint(clf.tree)


# create adjusted splitting criterion
# predictions = clf.predict(X_test)
# accuracy = print_scores(y_test, predictions['prediction'])
