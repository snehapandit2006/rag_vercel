from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def build_vectorstore(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)

    return db


def ask_rag(db, query):

    llm = ChatOpenAI(model="gpt-4o-mini")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k":3}),
        return_source_documents=True
    )

    result = qa({"query": query})

    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }