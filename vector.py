"""This module handles the creation of a vector database using Chroma and Ollama embeddings."""

import pathlib
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

DB_LOCATION = "./chroma_db"

def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)  # type: ignore

def get_embedding_model() -> OllamaEmbeddings:
    """Create embeddings for the specified text column in the DataFrame."""
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
    return embedding_model

def create_chroma_db(df: pd.DataFrame, embedding_model: OllamaEmbeddings) -> Chroma:
    """Create a Chroma database from the DataFrame and embedding model."""
    if not pathlib.Path(DB_LOCATION).exists():
        print("Creating Chroma database...")
        documents = [
            Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(idx),
            )
            for idx, row in df.iterrows()
        ]
        vector_store = Chroma.from_documents(  # type: ignore
            documents,
            embedding_model,
            collection_name="restaurant_reviews",
            persist_directory=DB_LOCATION,
        )
        return vector_store
    else:
        print("Loading existing Chroma database...")
        return Chroma(
            collection_name="restaurant_reviews",
            embedding_function=embedding_model,
            persist_directory=DB_LOCATION,
        )

def get_retriever():
    """Return the retriever for the Chroma database."""
    df = load_csv("documents/realistic_restaurant_reviews.csv")
    embedding_model = get_embedding_model()
    chroma_db = create_chroma_db(df, embedding_model)
    print("Chroma database is ready for use.")
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever
