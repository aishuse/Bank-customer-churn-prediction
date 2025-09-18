import os
import pickle
import mlflow

MLFLOW_TRACKING_URI = "http://ec2-34-201-147-159.compute-1.amazonaws.com:5000/"
MODEL_NAME = "churn_pred_model"
MODEL_VERSION = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts", "selected_features.pkl")

# Load selected features immediately
with open(ARTIFACTS_PATH, "rb") as f:
    selected_features = pickle.load(f)

_model = None

def get_model():
    global _model
    if _model is None:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        _model = mlflow.sklearn.load_model(model_uri)
    return _model
