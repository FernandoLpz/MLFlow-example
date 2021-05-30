import sys
from urllib.parse import urlparse
from numpy import mod

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

import mlflow
import mlflow.sklearn

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

    with mlflow.start_run():

        model = DecisionTreeModel(max_depth=max_depth)
        model.load_data()
        model.train()
        model.evaluate()

        mlflow.log_param("tree_depth", max_depth)
        mlflow.log_metric("precision", model.precision)
        mlflow.log_metric("recall", model.recall)
        mlflow.log_metric("accuracy", model.accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model.tree, "model", registered_model_name="MyDecisionTreeModel")
        else:
            mlflow.sklearn.log_model(model.tree, "model")
