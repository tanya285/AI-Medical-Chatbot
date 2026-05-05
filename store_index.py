
import os
from dotenv import load_dotenv

# Document loading
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector DB
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# LLM
from langchain_groq import ChatGroq

# Prompt + chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)


from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def load_pdf_files(path):
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

documents = load_pdf_files("data")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts_chunk = text_splitter.split_documents(documents)

print(f"Chunks created: {len(texts_chunk)}")

 #Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Pinecone setup
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Store vectors
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)