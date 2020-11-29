import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = json.loads(raw_data)['data']
    np_data = np.array(data)
    # Get a prediction from the model
    predictions = model.predict(np_data)
    # print the data and predictions (so they'll be logged!)
    log_text = 'Data:' + str(data) + ' - Predictions:' + str(predictions)
    print(log_text)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)
