import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(files):

    docs = []

    for file in files:

        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            docs.extend(loader.load())

        else:
            loader = TextLoader(temp_path)
            docs.extend(loader.load())

    return docs