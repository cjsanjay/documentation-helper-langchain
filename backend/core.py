import os
from typing import Any, List, Tuple, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone

import pinecone
import consts

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str, chat_history: List[Tuple[str, Any]]=[]) -> Any:
    """
    Run llm query

    :param chat_history:
    :param query:
    :return:
    """
    embeddings = OpenAIEmbeddings()
    doc_search = Pinecone.from_existing_index(index_name=consts.INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=doc_search.as_retriever(),
                                               return_source_documents=True)
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm("What is langchain?"))