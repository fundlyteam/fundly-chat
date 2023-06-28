import os
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from transformers import pipeline

# fix this
def set_huggingfacehub_api_token(token):
    """
    Set the Hugging Face Hub API token.

    Args:
        token (str): Hugging Face Hub API token.
    """
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

def load_faiss_index(file_path, embeddings):
    """
    Load the FAISS vector store from a file.

    Args:
        file_path (str): Path to the vector store file.
        embeddings: Embeddings object.

    Returns:
        langchain.FAISS: Loaded FAISS vector store.
    """
    return FAISS.load_local(file_path, embeddings)

def generate_relevant_docs(db, query):
    """
    Generate relevant documents using the FAISS vector store.

    Args:
        db (langchain.FAISS): FAISS vector store.
        query (str): Query for similarity search.

    Returns:
        list: Relevant documents based on the query.
    """
    return db.similarity_search(query)

def load_language_model(repo_id, model_kwargs):
    """
    Load a language model from Hugging Face Hub.

    Args:
        repo_id (str): Hugging Face model repository ID.
        model_kwargs (dict): Keyword arguments for the model.

    Returns:
        langchain.HuggingFaceHub: Loaded language model.
    """
    return HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)

def load_question_answering_chain(language_model, chain_type):
    """
    Load a question answering chain.

    Args:
        language_model (langchain.HuggingFaceHub): Language model.
        chain_type (str): Type of question answering chain.

    Returns:
        langchain.chains.QuestionAnsweringChain: Loaded question answering chain.
    """
    return load_qa_chain(language_model, chain_type=chain_type)

def generate_answer(chain, input_documents, question):
    """
    Generate an answer using the question answering chain.

    Args:
        chain (langchain.chains.QuestionAnsweringChain): Question answering chain.
        input_documents (list): Input documents for the chain.
        question (str): Question to be answered.

    Returns:
        str: Generated answer.
    """
    return chain.run(input_documents=input_documents, question=question)

def test_main():
    #set_huggingfacehub_api_token("HUGGINGFACEHUB_API_TOKEN")

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Load FAISS index
    db = load_faiss_index("faiss_index", embeddings)
    
    query = "What is computational social science"
    print("loaded db, got query: ", query)
    # Generate relevant documents
    relevant_docs = generate_relevant_docs(db, query)
    print(relevant_docs)

    # Load language model
    #language_model = load_language_model("google/flan-t5-xl", model_kwargs={"temperature": 0, "max_length": 512})

    # Load question answering chain
    #chain = load_question_answering_chain(language_model, chain_type="stuff")

    # Generate answer
    #answer = generate_answer(chain, input_documents=relevant_docs, question=query)
    #print(answer)

if __name__ == '__main__':
    test_main()
