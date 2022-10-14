import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion="entropy")
clf.fit(X_train, y_train)

tree.plot_tree(clf)
# plt.show()

# tree structure
n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node: node_id (0) and depth (0)
while len(stack) > 0:
    # 'pop' ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # if the left and right child of a node are not the same, we have a split node
    is_split_node = children_left[node_id] != children_right[node_id]
    # if a split node, append left and right children and depth to stack
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print(
    "The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes)
    )
for i in range(n_nodes):
    print("node: {node}".format(node=i))
    if is_leaves[i]:
        print(
            "{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i
            )
        )
    else:
        print(
            "{space}node={node} is a split node: "
            "go to node {left} if X[:, {feature}] <= {threshold} "
            "else to node {right}.".format(
                space=node_depth[i] * "\t",
                node=i,
                left=children_left[i],
                feature=feature[i],
                threshold=threshold[i],
                right=children_right[i]
            )
        )

# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier as Dt
#
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
#
# clf = tree.DecisionTreeClassifier(criterion="entropy")
# clf = clf.fit(X, Y)
#
# clf2 = Dt(criterion="entropy")
# clf2 = clf2.fit(X, Y)
#
# tree.plot_tree(clf)
# tree.plot_tree(clf2)
