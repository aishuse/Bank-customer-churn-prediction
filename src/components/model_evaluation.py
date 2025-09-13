import os
import json
import joblib
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models import infer_signature

os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.us-east-1.amazonaws.com'

# Logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        logger.debug("Model loaded from %s", model_path)
        return model
    except Exception as e:
        logger.error("Error loading model from %s: %s", model_path, e)
        raise

def load_features(features_path: str):
    try:
        features = joblib.load(features_path)
        logger.debug("Selected features loaded from %s", features_path)
        return features
    except Exception as e:
        logger.error("Error loading features from %s: %s", features_path, e)
        raise

def load_data(file_path: str, features: list, target_col: str = "Attrition_Flag"):
    try:
        df = pd.read_csv(file_path)
        X = df[features]
        y = df[target_col]
        logger.debug("Test data loaded and filtered with selected features")
        return X, y
    except Exception as e:
        logger.error("Error loading test data: %s", e)
        raise

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug("Model evaluation completed")
        return report, cm, y_pred
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise

def log_confusion_matrix(cm, dataset_name):
    cm_file_path = "artifacts/confusion_matrix.png"
    os.makedirs(os.path.dirname(cm_file_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(cm_file_path)
    plt.close()
    mlflow.log_artifact(cm_file_path)

def save_metrics(report: dict, file_path: str):
    try:
        with open(file_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.debug(f"Evaluation metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {file_path}: {e}")
        raise

def save_model_info(run_id: str, model_path: str, file_path: str):
    try:
        # model_path should be just "best_model"
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving model info: %s", e)
        raise

def main():
    mlflow.set_tracking_uri("http://ec2-34-201-147-159.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("churn-model-evaluation")

    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            artifacts_dir = os.path.join(root_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)

            # Load model + features
            model = load_model(os.path.join(artifacts_dir, "best_model.pkl"))
            features = load_features(os.path.join(artifacts_dir, "selected_features.pkl"))

            # Load test data
            test_data_path = os.path.join(root_dir, "data/prepared/test_prepared.csv")
            X_test, y_test = load_data(test_data_path, features)

            # Evaluate model
            report, cm, y_pred = evaluate_model(model, X_test, y_test)

            # Save metrics for DVC and MLflow tracking
            metrics_path = os.path.join(artifacts_dir, "eval_metrics.json")
            save_metrics(report, metrics_path)
            mlflow.log_artifact(metrics_path)

            # Infer MLflow signature
            input_example = pd.DataFrame(X_test[:5], columns=features)
            signature = infer_signature(input_example, model.predict(X_test[:5]))

            # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                "best_model",
                signature=signature,
                input_example=input_example,
            )

            # Save model info JSON (correct: "best_model" only), and log to MLflow
            model_info_path = os.path.join(artifacts_dir, "experiment_info.json")
            save_model_info(run.info.run_id, "best_model", model_info_path)
            mlflow.log_artifact(model_info_path)

            # Log classification report metrics
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics["precision"],
                        f"test_{label}_recall": metrics["recall"],
                        f"test_{label}_f1-score": metrics["f1-score"],
                        f"test_{label}_support": metrics["support"]
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test")

            # Add metadata tags
            mlflow.set_tag("model_type", "LightGBM/XGBoost")
            mlflow.set_tag("task", "Churn Prediction")
            mlflow.set_tag("dataset", "Bank Customer Data")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
