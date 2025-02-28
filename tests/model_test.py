import json
from mlflow.tracking import MlflowClient
import mlflow
import os
import unittest 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN is not set")
os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
dagshub_url = 'https://dagshub.com'
repo_owner = "TruongThuyLiem"
repo_name = "ci_mlops"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final_model")
model_name = "Best Model"

class TestModelLoading(unittest.TestCase):
    def test_model_in_staging(self):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages = ['Staging'])
        self.assertGreater(len(versions), 0, "No model in staging")

    def test_model_loading(self):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages = ['Staging'])
        if not versions:
            self.fail("No model in staging")
        latest_version = versions[0].version
        run_id = versions[0].run_id
        logged_model = f"runs:/{run_id}/{model_name}"
        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Fail to load model: {e}")
        self.assertIsNotNone(loaded_model, "Model is None")
        print (f"Model successfully loaded from {logged_model}.")
    
    def test_mode_performence(self):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages = ['Staging'])
        if not versions:
            self.fail("No model in staging")
        run_id = versions[0].run_id
        logged_model = f"runs:/{run_id}/{model_name}"        
        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Fail to load model: {e}")
        data = pd.read_csv("data/processed/test_processed_median.csv")
        X = data.drop(columns=['Potability'], axis=1)
        y_test = data['Potability']
        y_pred = loaded_model.predict(X)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        self.assertGreaterEqual(acc, 0.3, "acc < 0.3")
        self.assertGreaterEqual(precision, 0.3, "precision < 0.3")
        self.assertGreaterEqual(recall, 0.3, "recall < 0.3")
        self.assertGreaterEqual(f1, 0.3, "f1 < 0.3")

if __name__ == "__main__":
    unittest.main()