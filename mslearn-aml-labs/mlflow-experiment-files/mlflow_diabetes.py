from azureml.core import Run
import pandas as pd
import mlflow

# start the MLflow experiment
with mlflow.start_run():
       
    # Load data
    data = pd.read_csv('diabetes.csv')

    # Count the rows and log the result
    row_count = (len(data))
    print('observations:', row_count)
    mlflow.log_metric('observations', row_count)
