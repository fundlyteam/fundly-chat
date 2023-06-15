import langchain
from langchain import FAISS
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

db = FAISS.load_local("faiss_index", embeddings)
query = "What is computational social science"
relevant_docs = db.similarity_search(query)
print(relevant_docs)

#
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})

chain = load_qa_chain(llm, chain_type="stuff")

query = "What is computational social science"
relevant_docs = db.similarity_search(query)
chain.run(input_documents=relevant_docs, question=query)

#
from transformers import pipeline

generator = pipeline(model="mrm8488/t5-base-finetuned-common_gen")
generator(
    f"query: {query} , relevant docs: {relevant_docs}"
)
