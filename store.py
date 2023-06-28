import os
import PyPDF2
from PyPDF2 import PdfReader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_files(directory_path):
    """
    Get a list of PDF files in the specified directory.

    Args:
        directory_path (str): Path to the directory containing PDF files.

    Returns:
        list: List of PDF files in the directory.
    """
    pdf_files = [file for file in os.listdir(directory_path) if file.endswith('.pdf')]
    return pdf_files

def read_pdf_files(pdf_files, directory_path):
    """
    Read data from PDF files and return the concatenated text.

    Args:
        pdf_files (list): List of PDF files.
        directory_path (str): Path to the directory containing the PDF files.

    Returns:
        str: Concatenated text from the PDF files.
    """
    raw_text = ''
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
    return raw_text

def wrap_text_preserve_newlines(text, width=110):
    """
    Wrap the input text while preserving newlines.

    Args:
        text (str): Input text.
        width (int): Maximum width for each line.

    Returns:
        str: Wrapped text with preserved newlines.
    """
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def split_text(raw_text):
    """
    Split the raw text into chunks.

    Args:
        raw_text (str): Raw text to be split.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)
    return texts

def generate_embeddings(texts):
    """
    Generate embeddings for the given texts.

    Args:
        texts (list): List of text chunks.

    Returns:
        langchain.vectorstores.FAISS: FAISS vector store containing the embeddings.
    """
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    return db

def save_embeddings(db, file_path):
    """
    Save the FAISS vector store to a file.

    Args:
        db (langchain.vectorstores.FAISS): FAISS vector store.
        file_path (str): Path to save the vector store file.
    """
    db.save_local(file_path)

def store_files():
    directory_path = 'documents'

    # Get PDF files
    pdf_files = get_pdf_files(directory_path)

    # Read data from PDF files
    raw_text = read_pdf_files(pdf_files, directory_path)

    # Wrap text with preserved newlines
    wrapped_text = wrap_text_preserve_newlines(raw_text)

    # Split text into chunks
    texts = split_text(wrapped_text)

    # Generate embeddings
    db = generate_embeddings(texts)

    # Save embeddings to a file
    save_embeddings(db, "faiss_index")

if __name__ == '__main__':
    store_files()
