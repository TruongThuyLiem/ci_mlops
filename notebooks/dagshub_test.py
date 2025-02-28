import dagshub
dagshub.init(repo_owner='TruongThuyLiem', repo_name='ci_mlops', mlflow=True)

import mlflow
mlflow.set_tracking_uri("https://dagshub.com/TruongThuyLiem/ci_mlops.mlflow")
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)