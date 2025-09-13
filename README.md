# 🏦 **Bank Customer Churn Prediction**

A **full ML lifecycle project** for predicting **customer churn** in banking, covering **data ingestion, feature engineering, model training, evaluation, deployment**, and a **real-time prediction interface**.  

**Live Prediction App:** [🔗 **Click Here**](http://ec2-34-201-147-159.compute-1.amazonaws.com:8080/)

---

## **Project Overview**

### 1️⃣ **Data Pipeline (DVC)**
- **Stages:** **Ingestion → Transformation → Preparation → Training → Evaluation → Registration**  
- Produces **artifacts** that feed the next step  
- Ensures **reproducibility, traceability, and version control**

### 2️⃣ **Model Building**
- Handles class imbalance with **SMOTE**  
- **Feature selection:** **RFECV**  
- Models trained: **LightGBM** & **XGBoost** → best model selected  
- **Hyperparameter tuning** & **early stopping**  
- **Evaluation metrics:** **Accuracy, Precision, Recall, F1-score, ROC-AUC**  
- **Artifacts saved:** **model, selected features, evaluation metrics**

### 3️⃣ **MLflow Model Registry**
- Tracks **experiments & metrics**  
- Models stored in **S3**  
- Supports **versioning, staging, production workflow**  
- Models can be **dynamically loaded** for prediction

### 4️⃣ **Deployment**
- **FastAPI** serves predictions via **REST API**  
- Supports **dynamic model loading** from **MLflow**  
- **Dockerized** for consistent deployment environments  
- **CI/CD automated** using **GitHub Actions**

### 5️⃣ **Frontend**
- **Streamlit** interface for **user-friendly input**  
- Calls **FastAPI endpoint** for **predictions**  
- Displays **predicted class & probability**

### 6️⃣ **Logging & Evaluation**
- **Comprehensive logging** for all stages  
- **Confusion matrices & metrics** tracked in **MLflow**  
- Ensures **transparent and reproducible evaluation**

---

## **Technologies**
- 🐍 **Python, LightGBM, XGBoost** – Model development  
- 📊 **DVC** – Data pipeline & reproducibility  
- 📈 **MLflow** – Experiment tracking & model registry  
- 🐳 **Docker** – Containerization  
- ⚙️ **GitHub Actions** – CI/CD deployment  
- 🚀 **FastAPI** – Backend API  
- 🖥️ **Streamlit** – Frontend UI
