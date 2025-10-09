from typing import Annotated, Sequence, TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# Set a browser-like user agent
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"

load_dotenv()

# === Model and Embeddings ===
model_ = os.environ.get("GROQ_MODEL")
model = ChatGroq(model=model_)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# List of URLs to scrape
# urls = [
#     "https://www.scotiabank.com/ca/en/personal/bank-accounts/chequing-accounts/package-features.html",
#     "https://www.scotiaitrade.com/en/home/learning-centre/faqs.html",
#     "https://help.scotiabank.com/",
#     "https://tc.scotiabank.com/personal/day-to-day-banking/frequently-asked-questions.html",
#     "https://www.scotiaitrade.com/en/home/learning-centre/faqs/new-clients.html",
#     "https://www.scotiawealthmanagement.com/ca/en/campaigns/swm-faqs.html"
# ]

# os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"

# documents = []

# for url in urls:
#     headers = {"User-Agent": os.environ["USER_AGENT"]}
#     response = requests.get(url, headers=headers)
#     response.raise_for_status()
    
#     soup = BeautifulSoup(response.text, "html.parser")
#     # Collapse whitespace
#     visible_text = " ".join(soup.get_text(separator="\n", strip=True).split())
    
#     doc = Document(page_content=visible_text, metadata={"source": url})
#     documents.append(doc)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from langchain_core.documents import Document
import time

# URLs to fetch
urls = [
    "https://www.scotiabank.com/ca/en/personal/bank-accounts/chequing-accounts/package-features.html",
    "https://www.scotiaitrade.com/en/home/learning-centre/faqs.html",
    "https://help.scotiabank.com/",
    "https://tc.scotiabank.com/personal/day-to-day-banking/frequently-asked-questions.html",
    "https://www.scotiaitrade.com/en/home/learning-centre/faqs/new-clients.html",
    "https://www.scotiawealthmanagement.com/ca/en/campaigns/swm-faqs.html"
]

# Selenium Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

documents = []

for url in urls:
    driver.get(url)
    time.sleep(3)  # wait for JS to render; adjust if needed
    
    # Expand dropdowns/collapsible sections (common for FAQ pages)
    try:
        buttons = driver.find_elements(By.XPATH, "//button | //a[@role='button']")
        for btn in buttons:
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.2)
    except:
        pass
    
    # Get full page text
    page_text = driver.find_element(By.TAG_NAME, "body").text
    page_text = " ".join(page_text.split())
    
    doc = Document(page_content=page_text, metadata={"source": url})
    documents.append(doc)

driver.quit()

print(f"Loaded {len(documents)} documents dynamically")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

doc_splits = text_splitter.split_documents(documents)

# Vector DB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# === Retriever Tool ===
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information from the Scotiabank chequing accounts page.",
)

tools = [retriever_tool]
llm_with_tool = model.bind_tools(tools)

# === Graph State ===
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    docs: list | None  # store retrieved docs

# === Nodes ===
def ai_assistant(state: State):
    """Decide whether to call retrieval or answer directly"""
    msg = state["messages"]
    query = msg[-1].content
    print("Message received:", query)

    # Decide if query is bank-related using a simple LLM classifier
    prompt = PromptTemplate.from_template(
        """Decide if the user's query is related to banking, bank accounts, or Scotiabank offers.

Query: {msg}

Respond only with 'bank' or 'other'."""
    )
    chain = prompt | model | StrOutputParser()
    decision = chain.invoke({"msg": query}).strip().lower()
    print("Query classification:", decision)

    # Trigger retrieval only if bank-related
    if decision == "bank":
        print("Bank-related query — retrieval triggered!")
        return {"messages": msg}  # retrieval will handle next
    else:
        print("Not bank-related — normal LLM response")
        prompt2 = PromptTemplate.from_template(
            "You are a helpful assistant. Answer clearly and naturally:\n\n{msg}"
        )
        chain2 = prompt2 | model | StrOutputParser()
        res = chain2.invoke({"msg": query})
        return {"messages": [AIMessage(content=res)]}


def retrieve_docs(state: State):
    """Retrieve context using retriever"""
    query = state["messages"][-1].content
    print("Retrieving documents for:", query)
    retrieved_docs = retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} docs")
    return {"messages": state["messages"], "docs": retrieved_docs}


def decide(state: State):
    """Decide if docs are relevant"""
    msg = state["messages"][-1].content
    docs = "\n\n".join([d.page_content for d in state["docs"]])
    prompt = PromptTemplate.from_template(
        "Decide whether the following documents are relevant to the question.\n\n"
        "Question: {msg}\n\nDocuments:\n{docs}\n\n"
        "Respond only with 'relevant' or 'irrelevant'."
    )
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"msg": msg, "docs": docs}).strip().lower()
    print("Relevance Decision:", response)
    if "relevant" in response:
        return "generator"
    else:
        return "rewriter"


def op_generator(state: State):
    msg = state["messages"][-1].content
    docs_text = "\n\n".join([d.page_content for d in state["docs"]])

    prompt = PromptTemplate.from_template(
        """You are a helpful assistant answering questions about Scotiabank offers.

Question:
{msg}

Context (from the official Scotiabank site):
{docs}

Guidelines:
- If the context contains the answer, use it directly.
- If the context does NOT contain the answer, provide a clear answer based on general knowledge, 
  but preface with: "Note: This information is NOT from the Scotiabank webpage."
- Be concise and avoid repetition.

Answer:"""
    )

    chain = prompt | model | StrOutputParser()
    res = chain.invoke({"msg": msg, "docs": docs_text})
    return {"messages": [AIMessage(content=res)]}




def transform_query(state: State):
    """Rewrite question for better retrieval"""
    msg = state["messages"][-1].content
    prompt = PromptTemplate.from_template(
        "Rewrite the following question to make it clearer for search.\n\nQuestion: {msg}\n\nReturn only the rewritten question."
    )
    chain = prompt | model | StrOutputParser()
    rewritten = chain.invoke({"msg": msg}).strip()
    print("Rewritten query:", rewritten)

    # After rewriting, run retrieval again automatically
    retrieved_docs = retriever.invoke(rewritten)
    docs_text = "\n\n".join([d.page_content for d in retrieved_docs])
    gen_prompt = PromptTemplate.from_template(
        """You are a helpful assistant answering strictly using Scotiabank’s official webpage text.

Rewritten Question:
{msg}

Context (from the official Scotiabank site):
{docs}

Guidelines:
- Base your answer ONLY on the context.
- If unavailable, say:
  "The information is not available in the Scotiabank webpage context."

Answer:"""
    )
    chain2 = gen_prompt | model | StrOutputParser()
    res = chain2.invoke({"msg": rewritten, "docs": docs_text})
    return {"messages": [AIMessage(content=res)]}


# === Graph Setup ===
g = StateGraph(State)
g.add_node("ai_assistant", ai_assistant)
g.add_node("retrieve", retrieve_docs)
g.add_node("op_generator", op_generator)
g.add_node("transform_query", transform_query)

# Conditional transitions
# Always go from ai_assistant → retrieve on first query
g.add_edge("ai_assistant", "retrieve")
g.add_conditional_edges("retrieve", decide, {"generator": "op_generator", "rewriter": "transform_query"})

# Normal edges
g.add_edge(START, "ai_assistant")
g.add_edge("op_generator", END)
g.add_edge("transform_query", END)  # end after rewriting + generating answer

# Compile app
app = g.compile()


def get_rag():
    return app
