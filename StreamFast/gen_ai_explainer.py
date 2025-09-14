from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
# from dotenv import load_dotenv
import os
# load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

class GenerativeAIExplainer:
    def __init__(self):
        # Initialize the Groq chat model
        self.client = ChatGroq(model="llama3-7b", api_key=api_key)

    def explain_prediction(self, customer_features: dict, prediction: int) -> str:
        prompt = f"""
        You are an AI assistant for a bank.
        A customer has the following features: {customer_features}.
        The churn model predicted: {"Churn" if prediction == 1 else "Not Churn"}.

        Explain in simple terms why the model made this prediction.
        Keep it short (3–5 sentences), business-friendly, and avoid jargon.
        """

        # Wrap prompt in a HumanMessage
        response = self.client([HumanMessage(content=prompt)])

        # The response is a string
        return response.content
