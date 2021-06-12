import os
import sys
import numpy as np
from urllib.parse import urlparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

DEFAULT_MAX_DEPTH = 2

class DecisionTreeModel:
    '''
    Description: Implementation of a classifier based on a decision tree for 
    the breast cancer dataset. This implementation accepts as its only input parameter, 
    the maximum depth of the tree whose default value is 2.
    '''

    def __init__(self, max_depth: int) -> None:
        self.max_depth = max_depth
        pass

    def load_data(self):
        '''
        Description: Load and split the breast cancer dataset
        '''

        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.33, random_state=42)

    def train(self):
        '''
        Description: Train a classifier based on a decision tree
        '''
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(self.x_train, self.y_train)

    def evaluate(self):
        '''
        Description: Calculate precision, recall and accurcy metrics
        '''
        self.y_pred = self.tree.predict(self.x_test)
        self.precision = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)


if __name__ == '__main__':

    # Validate input parameter that refers to the maximum depth of the tree
    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_DEPTH

    # Defines the MLFlow server IP
    remote_server_uri = "http://localhost"
    mlflow.set_tracking_uri(remote_server_uri)

    # For authentication with the server, define the credentials to use.
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'mlflow-user'
    os.environ['MLFLOW_TRACKING_PASSWORD']='f3rn4nd0.'

    # Creates an experiment where all runs will be organized in the server side
    mlflow.set_experiment("Experiment-Server")

    # Intialize a context for running mlflow tracking
    with mlflow.start_run():

        # Initialize the model, loads data, trains and evalutes.
        model = DecisionTreeModel(max_depth=max_depth)
        model.load_data()
        model.train()
        model.evaluate()

        # Log metrics and parameter
        mlflow.log_param("tree_depth", max_depth)
        mlflow.log_metric("precision", model.precision)
        mlflow.log_metric("recall", model.recall)
        mlflow.log_metric("accuracy", model.accuracy)

        # For explainability, an input example is created which will be logged in the MLmodel file
        input_example = np.random.rand(1,30)

        # Creates a signature
        signature = infer_signature(model.x_test, model.x_test)

        # Logs model and initialize a registered version called "Decision_Tree"
        mlflow.sklearn.log_model(model.tree, "MyModel-dt", registered_model_name="Decision_Tree", signature=signature, input_example=input_example)