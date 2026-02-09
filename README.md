# Ollama Chroma RAG 🍕

A simple Retrieval-Augmented Generation (RAG) example using **Ollama**, **LangChain**, and **Chroma** to answer questions based on restaurant (pizza) reviews stored in a CSV file.

This project demonstrates how to:
- Embed structured data (CSV) using Ollama embeddings
- Store and persist vectors in a Chroma database
- Retrieve relevant documents for a user query
- Generate grounded answers using a local LLM

---

## 🧠 Architecture Overview

CSV Reviews  
→ Ollama Embeddings (mxbai-embed-large)  
→ Chroma Vector Database (persistent)  
→ Retriever (Top-K documents)  
→ Ollama LLM (llama3.2)  
→ Answer to User Question

---

## 📁 Project Structure

```
.
├── vector.py
├── main.py
├── documents/
│   └── realistic_restaurant_reviews.csv
├── chroma_db/
└── README.md
```

---

## 🚀 Requirements

- Python 3.10+
- Ollama installed and running locally
- Required Ollama models:
  ```
  ollama pull llama3.2
  ollama pull mxbai-embed-large
  ```

---

## 📦 Installation

```
git clone https://github.com/samueladole/ollama-chroma-rag.git
cd ollama-chroma-rag

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Example requirements.txt:
```
langchain
langchain-core
langchain-chroma
langchain-ollama
pandas
```

---

## ▶️ Running the App

```
python main.py
```

Ask questions like:
- Which pizza place has the best reviews?
- What do customers say about the crust?
- Are there complaints about service?

---

## 🗄️ How It Works

1. Loads reviews from a CSV file  
2. Embeds text using Ollama embeddings  
3. Stores vectors in a persistent Chroma database  
4. Retrieves relevant reviews per question  
5. Generates answers using a local LLM  

---

## 🛠️ Customization Ideas

- Swap datasets (movies, products, support tickets)
- Add metadata filtering
- Change embedding or LLM models
- Add a web UI (Streamlit or FastAPI)

---

## 📄 License

MIT License
