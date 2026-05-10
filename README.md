# 🏥 AI Medical Chatbot

> A Retrieval-Augmented Generation (RAG) powered medical chatbot that answers health queries using curated medical literature — built with LangChain, Pinecone, Groq (Llama 3.1), and Flask.

---

## 📌 Overview

The **AI Medical Chatbot** is an intelligent conversational assistant designed to provide reliable, context-aware medical information. Rather than relying on a static FAQ database or generic LLM responses, it uses a **RAG pipeline** — retrieving relevant chunks from a medical knowledge base stored in Pinecone and passing them as context to a Llama 3.1 language model via Groq for fast, grounded answers.

This project combines the power of semantic search with modern LLMs to eliminate hallucinations and ensure responses are always anchored in real medical content.

---

## ✨ Features

- 🔍 **Retrieval-Augmented Generation** — answers are grounded in indexed medical documents, not guessed
- ⚡ **Groq + Llama 3.1** — ultra-fast LLM inference via the `llama-3.1-8b-instant` model
- 🧠 **HuggingFace Embeddings** — semantic similarity search across the medical knowledge base
- 📦 **Pinecone Vector Store** — scalable, cloud-hosted vector database for document retrieval
- 🌐 **Flask Web App** — clean REST API with a responsive chat frontend
- 🚀 **Render-ready Deployment** — configured for live hosting out of the box

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language Model | Groq API — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace (via `src/helper.py`) |
| Vector Store | Pinecone (`medical-chatbot` index) |
| Orchestration | LangChain (`retrieval_chain`, `stuff_documents_chain`) |
| Backend | Python 3.10 + Flask |
| Frontend | HTML / CSS / JavaScript |
| Deployment | Render |

---

## 📁 Project Structure

```
AI-Medical-Chatbot/
├── app.py                  # Flask app — routes and RAG chain setup
├── store_index.py          # Script to embed and index docs into Pinecone
├── setup.py                # Package setup
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for deployment
├── template.sh             # Project scaffolding script
├── src/
│   ├── helper.py           # HuggingFace embedding loader
│   └── prompt.py           # System prompt for the LLM
├── data/                   # Medical documents for indexing
├── research/               # Jupyter notebooks and experiments
├── static/                 # CSS, JS, images
└── templates/
    └── index.html          # Chat UI
```

---

## ⚙️ How It Works

```
User Query
    │
    ▼
HuggingFace Embedding Model
    │
    ▼
Pinecone Similarity Search (top-k=3 chunks)
    │
    ▼
LangChain Retrieval Chain
    │
    ▼
Groq LLM (Llama 3.1-8b-instant) + System Prompt
    │
    ▼
Medical Answer → Flask Response → Chat UI
```

1. The user types a medical question in the chat interface
2. The query is embedded using a HuggingFace model
3. Pinecone retrieves the 3 most semantically similar document chunks
4. LangChain assembles the context and query into a prompt
5. Groq runs inference with Llama 3.1 and returns a grounded answer
6. The answer is displayed in the chat UI

---

## 🚀 How to Run?

### Steps:

**STEP 00 — Clone the Repository**

```bash
git clone https://github.com/tanya285/AI-Medical-Chatbot.git
cd AI-Medical-Chatbot
```

**STEP 01 — Create a conda environment after opening the repository**

```bash
conda create -n mediassist python=3.10 -y
```

```bash
conda activate mediassist
```

**STEP 02 — Install the requirements**

```bash
pip install -r requirements.txt
```

**STEP 03 — Set up environment variables**

Create a `.env` file in the root directory and add your API keys:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

**STEP 04 — Index the medical knowledge base**

Place your medical PDF/text documents in the `data/` folder, then run:

```bash
python store_index.py
```

**STEP 05 — Run the app**

```bash
python app.py
```

Open your browser at: **http://localhost:10000**

---

## ☁️ Deployment (Render)

This app is configured for [Render](https://render.com/) deployment:

- The Flask app listens on `host="0.0.0.0"` and reads the `PORT` environment variable (defaults to `10000`)
- Set `PINECONE_API_KEY` and `GROQ_API_KEY` in your Render service's environment variables
- Set the start command to: `python app.py`

---

## ⚠️ Disclaimer

This chatbot is intended for **informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 🙋‍♀️ Author

**Tanya** — [@tanya285](https://github.com/tanya285)
