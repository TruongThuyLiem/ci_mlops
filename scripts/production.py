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

def promte_model_to_production():
    client = MlflowClient()

    staging_versions = client.get_latest_versions(model_name, stages=['Staging'])
    if not staging_versions:
        print ("No model in staging")
        return
    latest_staging_version = staging_versions[0]
    staging_version_number = latest_staging_version.version
    production_version = client.get_latest(model_name, stages= ['Production'])
    if production_version:
        current_production_version = production_version[0]
        production_version_number = current_production_version.version
        client.transition_model_version_stage(
            name=model_name,
            version = production_version_number,
            stage = 'Archived',
            archive_existing_versions=False
        )
        print (f"Archived model version {production_version_number} in Production")
    else:
        print ("No model in Production")
    client.transition_model_version_stage(
                name=model_name,
                version = staging_version_number,
                stage = 'Production',
                archive_existing_versions=False
            )
    print ("Promoted model version {staging_version_number} to Production")

if __name__ == "__main__":
    promte_model_to_production()
