import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever, load_csv

st.set_page_config(page_title="Pizza Reviews RAG", page_icon="🍕")

st.title("🍕 Pizza Reviews Assistant")
st.text("Ask questions about restaurant reviews using a local RAG pipeline.")


@st.cache_resource
def load_retriever():
    """Load the retriever for fetching relevant reviews."""
    return get_retriever()


@st.cache_resource
def load_llm():
    """Load the Ollama LLM for generating responses."""
    return OllamaLLM(model="llama3.2")


retriever = load_retriever()
llm = load_llm()

TEMPLATE = """
You are a helpful AI assistant who answers questions about pizza restaurants.

Here are some relevant reviews:
{reviews}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)
chain = prompt | llm  # type: ignore

question = st.text_input("Ask a question about the reviews:")

if question:
    with st.spinner("Thinking..."):
        reviews = retriever.invoke(question)
        response = chain.invoke(  # type: ignore
            {
                "reviews": reviews,
                "question": question,
            },
        )
    st.subheader("Answer")
    st.markdown(
        f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>{response}</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Dataset Information")
    df = load_csv("documents/realistic_restaurant_reviews.csv")
    st.write(df) # type: ignore
