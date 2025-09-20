import sys
import os

# Ensure /app is in the Python path so StreamFast can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Annotated, Literal, Sequence, TypedDict
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
import streamlit as st
import os
from StreamFast.model_loader import get_model, selected_features

model = get_model()  # only loads when needed

# Your FastAPI app code here

# ==========================
# Load Environment & Model
# ==========================
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
groq_model = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
# ==========================
# State Definition
# ==========================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    predictions: pd.DataFrame
    churned_customers: pd.DataFrame
    churned_list: list[dict]

# ==========================
# Utility Functions
# ==========================
def sendmail(to_email: str, subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "aamikutty180@gmail.com"
    msg["To"] = to_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(msg["From"], [msg["To"]], msg.as_string())

# ==========================
# Data & Model Loading
# ==========================
# model_path = "lgbm_model.pkl"
# selected_features_path = "selected_features.pkl"
loaded_model = model
# selected_features = joblib.load(selected_features_path)

loader = PyPDFLoader("data/csv/offer_Guide.pdf")
docs = loader.load()

# ==========================
# Core Functions
# ==========================
def usermsg(state: AgentState):
    return {"messages": state['messages']}

def decide_check(state: AgentState) -> Literal["predict_churners", "other_query"]:
    msg = state['messages']
    prompt = PromptTemplate(
        template="""
        You are a deciding assistant.

        Task:
        Decide if the user's message is about identifying churn risk, churners, 
        customers leaving a bank, or customers about to leave.

        Rules:
        - If the message is related to churn, output ONLY: churn
        - Otherwise, output ONLY: other

        User message: {msg}
        """,
        input_variables=["msg"]
    )
    chain = prompt | groq_model
    response = chain.invoke({"msg": msg})
    response_text = response.content.strip().lower()
    return "predict_churners" if response_text == "churn" else "other_query"

def predict_churn_selected_features(raw_df: pd.DataFrame):
    df = raw_df.copy()

    if "Customer_ID" in df.columns:
        df = df.drop(columns=["Customer_ID"])
    
    # Engineer only selected features
    if 'Avg_Transaction_Amount' in selected_features:
        df['Avg_Transaction_Amount'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct']

    # One-hot encode Marital_Status if needed
    marital_dummies = [col for col in selected_features if col.startswith('Marital_Status_')]
    df_encoded = pd.get_dummies(df, columns=['Marital_Status']) if marital_dummies else df.copy()

    # Align columns with selected_features
    df_aligned = df_encoded.reindex(columns=selected_features, fill_value=0)

    # Predict
    prob = loaded_model.predict_proba(df_aligned.values)[:, 1]
    pred_class = (prob >= 0.5).astype(int)

    return pd.DataFrame({
        "Predicted_Class": pred_class,
        "Churn_Probability": prob
    })

def predict_churners(state: AgentState):
    customers_df = pd.read_csv("data/csv/customers.csv")
    details_df = pd.read_csv("data/csv/custdetails.csv")  

    if "Customer_ID" not in customers_df.columns:
        customers_df.insert(0, "Customer_ID", range(1, len(customers_df) + 1))

    predictions = predict_churn_selected_features(customers_df)
    results = pd.concat([customers_df, predictions], axis=1)

    churned_customers = results[results["Predicted_Class"] == 1]
    churned_with_details = churned_customers.merge(details_df, on="Customer_ID", how="left")
    churned_list = churned_with_details.to_dict(orient="records")

    state["predictions"] = results
    state["churned_customers"] = churned_with_details
    state["churned_list"] = churned_list
    return state

def generate_email(profile: dict):
    parser = StrOutputParser()
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful banking AI assistant. "
            "You are given a customer profile and a retention offers guide. "
            "Your task: generate ONLY the personalized email body to send to the customer. "
            "Do not include subject line. End with 'Best regards, Customer Service Team'. "
            "Do not include analysis or explanation. The tone should be personal and professional."
        ),
        HumanMessagePromptTemplate.from_template(
            "Customer profile:\n{profile}\n\nGuide:\n{guide}\n\nNow write the email body only."
        )
    ])
    chain = chat_template | groq_model | parser
    return chain.invoke({"guide": docs, "profile": str(profile)})

def send_email(state: AgentState):
    for cust in state["churned_list"]:
        email_body = generate_email(cust)
        sendmail(
            to_email=cust["Email"],
            subject="We Value You! Exclusive Offers to Reward Your Loyalty",
            body=email_body
        )
        print(f"✅ Email sent to {cust['Full_Name']} at {cust['Email']}")
    return state

def other_query(state: AgentState):
    res = groq_model.invoke(state['messages'])
    return {"messages": state["messages"] + [res]}

# ==========================
# Workflow Graph
# ==========================


def get_ret_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("usermsg", usermsg)
    workflow.add_node("predict_churners", predict_churners)
    workflow.add_node("send_email", send_email)
    workflow.add_node("other_query", other_query)

    workflow.add_edge(START, "usermsg")
    workflow.add_conditional_edges("usermsg", decide_check, {
        "predict_churners": "predict_churners",
        "other_query": "other_query"
    })
    workflow.add_edge("predict_churners", "send_email")
    workflow.add_edge("send_email", END)

    ret_app = workflow.compile()
    return ret_app
