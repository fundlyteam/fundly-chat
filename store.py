import langchain
import os
import PyPDF2
# import docx

#pdf_path = 'fia_2023_formula_1_sporting_regulations_-_issue_2_-_2022-09-30.pdf'  # Replace with the actual path to your PDF file
from PyPDF2 import PdfReader
# Directory path containing the PDF files
directory_path = 'documents'

# List all the PDF files in the directory
pdf_files = [file for file in os.listdir(directory_path) if file.endswith('.pdf')]
docx_files = [file for file in os.listdir(directory_path) if file.endswith('.docx')]


# Read data from the PDF files
raw_text = ''
for pdf_file in pdf_files:
    pdf_path = os.path.join(directory_path, pdf_file)
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

# # Read data from the Word files
# for docx_file in docx_files:
#     docx_path = os.path.join(directory_path, docx_file)
#     doc = Document(docx_path)
#     for paragraph in doc.paragraphs:
#         text = paragraph.text
#         if text:
#             raw_text += text

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

from langchain.vectorstores import FAISS

db = FAISS.from_texts(texts, embeddings)
db.save_local("faiss_index")

