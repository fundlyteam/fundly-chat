import streamlit as st
from store import wrap_text_preserve_newlines, split_text, generate_embeddings, save_embeddings
from test import test_main, generate_relevant_docs
import PyPDF2
from PyPDF2 import PdfReader
import pickle


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    
def main():
    st.header("Chat with PDF 💬")
    
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
        
        # Wrap text with preserved newlines
        wrapped_text = wrap_text_preserve_newlines(raw_text)

        # Split text into chunks
        texts = split_text(wrapped_text)

        # Generate embeddings
        db = generate_embeddings(texts)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # Save embeddings to a file
        with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(db, f)

    query = st.text_input("Ask questions about your PDF file:")

    if query:
        relevant_docs = generate_relevant_docs(db, query)
        st.write(relevant_docs)


if __name__ == '__main__':
    main()
