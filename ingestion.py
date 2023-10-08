import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

import pinecone
import consts

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest")
    documents = loader.load()
    print(f"Documents loaded: {len(documents)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    document_chunks = text_splitter.split_documents(documents=documents)
    print(f"Splitted documents: {len(document_chunks)}")

    for doc in document_chunks:
        old_url = doc.metadata["source"]
        new_url = old_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Inserting in pinecone: {len(document_chunks)}")

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents=document_chunks, embedding=embeddings, index_name=consts.INDEX_NAME)
    print("Added the documents to the vector store")


if __name__ == "__main__":
    ingest_docs()
