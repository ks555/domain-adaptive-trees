from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as Dt

X = [[0, 0], [1, 1]]
Y = [0, 1]

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X, Y)

clf2 = Dt(criterion="entropy")
clf2 = clf2.fit(X, Y)

tree.plot_tree(clf)
tree.plot_tree(clf2)
