import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

MAX_DEPTH = 2

if __name__ == '__main__':

    if len(sys.argv) > 1:
        max_depth = int(sys.argv[1])
    else:
        max_depth = MAX_DEPTH

    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(x_train, y_train)

    y_pred = tree.predict(x_test)

    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")