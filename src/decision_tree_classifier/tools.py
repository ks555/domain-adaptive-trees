from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def split_data(X: DataFrame, y: Series, size: float = 0.5, rs: int = 123):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=rs)
    return x_train, x_test, y_train, y_test


def print_scores(y_test: Series, y_pred: Series):
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    return accuracy_score(y_test, y_pred)
