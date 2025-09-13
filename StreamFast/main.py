from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List, Literal
import mlflow

# ---------------------------
# MLflow setup
# ---------------------------
MLFLOW_TRACKING_URI = "http://ec2-34-201-147-159.compute-1.amazonaws.com:5000/"
MODEL_NAME = "churn_pred_model"
MODEL_VERSION = "1"  # or "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

# Load selected features
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "artifacts", "selected_features.pkl"), "rb") as f:
    selected_features = pickle.load(f)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Churn Prediction API", version="1.0")

# ---------------------------
# Input schema
# ---------------------------
class CustomerInput(BaseModel):
    Customer_Age: int
    Gender: Literal[0, 1]
    Dependent_count: int
    Education_Level: Literal["High School", "Graduate", "Uneducated", "Unknown", "College", "Post-Graduate", "Doctorate"]
    Marital_Status: Literal["Married", "Single", "Unknown", "Divorced"]
    Income_Category: Literal["$60K - $80K", "Less than $40K", "$80K - $120K", "$40K - $60K", "$120K +", "Unknown"]
    Card_Category: Literal["Blue", "Gold", "Silver", "Platinum"]
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: int
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float

# ---------------------------
# Prediction function
# ---------------------------
def predict_churn_selected_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if 'Avg_Transaction_Amount' in selected_features:
        df['Avg_Transaction_Amount'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct']

    marital_dummies = [col for col in selected_features if col.startswith('Marital_Status_')]
    df_encoded = pd.get_dummies(df, columns=['Marital_Status']) if marital_dummies else df.copy()
    df_aligned = df_encoded.reindex(columns=selected_features, fill_value=0)

    X_new = df_aligned.values
    prob = model.predict_proba(X_new)[:, 1]
    pred_class = (prob >= 0.5).astype(int)

    return pd.DataFrame({
        "Predicted_Class": pred_class,
        "Churn_Probability": prob
    })

# ---------------------------
# API endpoint
# ---------------------------
@app.post("/predict")
def predict(customers: List[CustomerInput]):
    input_df = pd.DataFrame([c.dict() for c in customers])
    predictions = predict_churn_selected_features(input_df)
    return predictions.to_dict(orient="records")

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
