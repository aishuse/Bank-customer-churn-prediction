# 🏦 Bank Customer Churn Prediction with Generative AI

A **full ML lifecycle project** for predicting **customer churn** in banking, covering **data ingestion, feature engineering, model training, evaluation, deployment**, and a **real-time prediction interface** with **AI-driven explanations**.  

This project also includes a **bulk churn detection workflow** that can automatically **generate and send personalized retention emails** using **LangChain + Groq**, orchestrated with **LangGraph** for **stateful AI workflow management**.


**Live Prediction App:** [🔗 Click Here](http://ec2-52-91-173-215.compute-1.amazonaws.com:8080/)

---

## Project Overview

### 1️⃣ Data Pipeline (DVC)
- **Stages:** Ingestion → Transformation → Preparation → Training → Evaluation → Registration  
- Produces **artifacts** that feed the next step  
- Ensures **reproducibility, traceability, and version control**

### 2️⃣ Model Building
- Handles class imbalance with **SMOTE**  
- **Feature selection:** **RFECV**  
- Models trained: **LightGBM** & **XGBoost** → best model selected  
- **Hyperparameter tuning** & **early stopping**  
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC  
- **Artifacts saved:** model, selected features, evaluation metrics

### 3️⃣ MLflow Model Registry
- Tracks **experiments & metrics**  
- Models stored in **S3**  
- Supports **versioning, staging, production workflow**  
- Models can be **dynamically loaded** for prediction

### 4️⃣ Deployment
- **FastAPI** serves predictions via **REST API**  
- Supports **dynamic model loading** from MLflow  
- **Dockerized** for consistent deployment environments  
- **CI/CD automated** using GitHub Actions

### 5️⃣ Frontend
- **Streamlit** interface for **user-friendly input**  
- Calls **FastAPI endpoint** for predictions  
- Displays **predicted class & probability**  
- **Generative AI explanation:**  
  - Uses **LangChain + Groq** to generate **business-friendly explanations**  
  - Explains **why the model predicted churn** for each customer  
  - Helps **non-technical users understand model decisions**

### 6️⃣ Bulk Churn Detection & Retention Emails
- **Detect churners in bulk** using historical customer data  
- **Generates personalized retention emails** automatically for each churner  
- Emails use content from **retention offer guide PDF**  
- Emails are sent via **Gmail SMTP**  
- Orchestrated with **LangGraph R8** workflow:
  - `usermsg` → checks user intent
  - `predict_churners` → predicts churn using ML model
  - `send_email` → generates & sends emails
  - `other_query` → handles non-churn queries  
- **Streamlit button workflow**:  
  1. Identify churned customers  
  2. Display results in a **dataframe**  
  3. Generate & send **personalized emails** automatically  

### 7️⃣ Logging & Evaluation
- **Comprehensive logging** for all stages  
- **Confusion matrices & metrics** tracked in MLflow  
- Ensures **transparent and reproducible evaluation**
---

### 8️⃣ Retrieval-Augmented Generation (RAG) for Banking FAQ Assistant

An integrated **RAG-based chatbot** that intelligently answers **bank-related FAQs** using **live website data**.  
The chatbot:
- Dynamically **scrapes** official bank FAQ and offer pages using **Selenium**  
- Splits and **embeds content** using `HuggingFace Embeddings` (`all-MiniLM-L6-v2`)  
- Stores vectors in a **Chroma vector database**  
- Uses **LangChain + LangGraph + ChatGroq** to:
  - Classify queries  
  - Retrieve relevant content  
  - Decide contextual relevance  
  - Generate **concise, factual, and context-based answers**  

## Technologies
- 🐍 **Python, LightGBM, XGBoost** – Model development  
- 📊 **DVC** – Data pipeline & reproducibility  
- 📈 **MLflow** – Experiment tracking & model registry  
- 🐳 **Docker** – Containerization  
- ⚙️ **GitHub Actions** – CI/CD deployment  
- 🚀 **FastAPI** – Backend API  
- 🖥️ **Streamlit** – Frontend UI  
- 🤖 **LangChain + Groq** – AI-powered model explanations & email generation  
- 📌 **LangGraph** – Orchestrates AI workflow & stateful decision logic  