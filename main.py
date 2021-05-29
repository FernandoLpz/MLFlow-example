import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

DEFAULT_MAX_DEPTH = 2

class DecisionTreeModel:

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        pass

    def load_data(self):
        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.33, random_state=42)

    def train(self):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(self.x_train, self.y_train)

    def evaluate(self):
        y_pred = self.tree.predict(self.x_test)
        self.precision = precision_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.accuracy = accuracy_score(self.y_test, y_pred)


if __name__ == '__main__':

    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_DEPTH

    model = DecisionTreeModel(max_depth=max_depth)