import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain.globals import set_debug
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from ollama import embeddings

set_debug(True)
import streamlit as st



llm = OllamaEmbeddings(model="llama3.2")
# PERPLEXITY_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = PerplexityEmbeddings(api_key=PERPLEXITY_API_KEY)


document = TextLoader('job_listing.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                               chunk_overlap=10)

chunks = text_splitter.split_documents(document)
db=Chroma.from_documents(chunks, llm)
retreiver = db.as_retriever()

text = input("Enter the query: ")
docs = retreiver.invoke(text)

# for doc in docs:
print(docs[0].page_content)