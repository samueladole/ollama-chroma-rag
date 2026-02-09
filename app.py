import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever

st.set_page_config(page_title="Pizza Reviews RAG", page_icon="🍕")

st.title("🍕 Pizza Reviews Assistant")
st.text("Ask questions about restaurant reviews using a local RAG pipeline.")

@st.cache_resource
def load_retriever():
    return get_retriever()

@st.cache_resource
def load_llm():
    return OllamaLLM(model="llama3.2")

retriever = load_retriever()
llm = load_llm()

template = '''
You are a helpful AI assistant who answers questions about pizza restaurants.

Here are some relevant reviews:
{reviews}

Question:
{question}
'''

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm # type: ignore

question = st.text_input("Ask a question about the reviews:")

if question:
    with st.spinner("Thinking..."):
        reviews = retriever.invoke(question)
        response = chain.invoke({ # type: ignore
            "reviews": reviews,
            "question": question
        })
    st.subheader("Answer")
    st.text(response)
