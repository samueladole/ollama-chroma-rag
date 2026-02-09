# Ollama Chroma RAG 🍕

A simple Retrieval-Augmented Generation (RAG) example using **Ollama**, **LangChain**, **Chroma**, and **Streamlit** to answer questions based on restaurant (pizza) reviews stored in a CSV file.

The project supports:
- A **CLI chat app**
- A **Streamlit web UI**
- Fully local inference using Ollama

---

## 🧠 Architecture Overview

CSV Reviews  
→ Ollama Embeddings (`mxbai-embed-large`)  
→ Chroma Vector Database (persistent)  
→ Retriever (Top-K documents)  
→ Ollama LLM (`llama3.2`)  
→ Answer to User Question  

---

## 📁 Project Structure

```
.
├── vector.py              # Loads CSV, creates/loads Chroma DB, returns retriever
├── main.py                # CLI-based chat app
├── app.py                 # Streamlit web UI
├── documents/
│   └── realistic_restaurant_reviews.csv
├── chroma_db/             # Persisted Chroma vector store
└── README.md
```

---

## 🚀 Requirements

- Python **3.13+**
- [Ollama](https://ollama.com/) installed and running locally
- Required Ollama models:
  ```bash
  ollama pull llama3.2
  ollama pull mxbai-embed-large
  ```

---

## 📦 Installation

```bash
git clone https://github.com/samueladole/ollama-chroma-rag.git
cd ollama-chroma-rag

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Example `requirements.txt`:
```txt
langchain
langchain-core
langchain-chroma
langchain-ollama
pandas
streamlit
```

---

## ▶️ Running the CLI App

```bash
python main.py
```

Ask questions like:
- Which pizza place has the best reviews?
- Are there any complaints about service?
- What do people say about the crust?

---

## 🖥️ Running the Streamlit App

```bash
streamlit run app.py
```

Then open:
```
http://localhost:8501
```

### Streamlit Features
- Cached retriever and LLM for fast responses
- Simple text input interface
- Uses the same RAG pipeline as the CLI app

---

## 🗄️ How It Works

1. Reviews are loaded from a CSV file
2. Text is embedded using Ollama embeddings
3. Vectors are stored in a persistent Chroma database
4. Relevant reviews are retrieved per user question
5. A local LLM generates a grounded response

The Chroma database is created once and reused on subsequent runs.

---

## 🛠️ Customization Ideas

- Add metadata filtering (rating, date)
- Show retrieved reviews in the UI
- Enable streaming token responses
- Swap datasets (movies, products, support tickets)
- Add Docker support

---

## ⚠️ Notes

- Runs **entirely locally**
- No external APIs required
- Ideal for experimenting with private data and RAG pipelines

---

## 📄 License

MIT License
