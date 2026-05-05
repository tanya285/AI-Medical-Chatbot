from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get API keys from environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing API keys. Set PINECONE_API_KEY and GROQ_API_KEY in environment variables.")

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# LLM
chatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Chains
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("index.html")  # make sure templates/index.html exists

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")

    if not msg:
        return "No input provided"

    try:
        response = rag_chain.invoke({"input": msg})
        return str(response["answer"])
    except Exception as e:
        return f"Error: {str(e)}"

# Run app (Render-compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # IMPORTANT for Render
    app.run(host="0.0.0.0", port=port)
