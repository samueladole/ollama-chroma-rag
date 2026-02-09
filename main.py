"""
This script demonstrates how to use the OllamaLLM
from the langchain_ollama library to generate a response to a chat prompt.
It defines a function to create an instance of
the OllamaLLM and another function to run a prompt and print the response.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever


def get_ollama_llm(model_name: str = "llama3.2") -> OllamaLLM:
    """Create an instance of the OllamaLLM with the specified model name."""
    llm = OllamaLLM(model=model_name)
    return llm


def main():
    """Run a prompt using the OllamaLLM and print the response."""

    llm = get_ollama_llm("llama3.2")

    template = """
    You are a helpful AI assistant bot who is an expert in answering questions about a pizza in.

    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}
    """

    # Create a chat prompt template
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm # type: ignore

    retriever = get_retriever()

    while True:
        print('\n---------------------------------')
        input_question = input("Ask your question (or type 'q' to quit): ")
        if input_question.lower() == 'q':
            print("Exiting the program. Goodbye!")
            break
        print('---------------------------------')

        # Get relevant documents using the retriever and generate a response using the chain
        reviews = retriever.invoke(input_question)
        response = chain.invoke({"reviews": reviews, "question": input_question}) # type: ignore
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
