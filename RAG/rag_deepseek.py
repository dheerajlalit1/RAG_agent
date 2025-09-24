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
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import uuid

from ollama import embeddings
from tenacity import wait_chain

from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# set_debug(True)
import streamlit as st

from typing import TypedDict

class MyState(TypedDict):
    input: str
    answer: str

embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
llm = ChatOllama(model="deepseek-r1:8b")



document = TextLoader('family.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                               chunk_overlap=20)

chunks = text_splitter.split_documents(document)
vector_store=Chroma.from_documents(chunks, embeddings)
retreiver = vector_store.as_retriever()

promt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert on family trees. Parse relationships (parent, child, spouse, sibling)"
               " from context and answer concisely. If context is missing, say 'I don't know'. {context}"),
     MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, retreiver, promt_template)

qa_chain = create_stuff_documents_chain(llm, promt_template)
rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id:history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)
#
# st.title("Lalit Family Tree")
# text = st.text_input("Enter the query: ")
# if text:
#     response = chain_with_history.invoke({"input": text},
#                                           {"configurable":{"session_id":"abc123"}})
#     st.write(response['answer'])


graph = StateGraph(MyState)
graph.add_node("qa", chain_with_history)
graph.set_entry_point("qa")


# 7. Add memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 8. Run agent
# config = {"configurable": {"session_id": "abc123"}}  # unique session id

st.title("Lalit Family Tree")
question = st.text_input("Ask me: ")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if question:
    result = app.invoke({"input": question},
                        config={"configurable":{"session_id": st.session_state.session_id,
                                                "thread_id": st.session_state.session_id}})
    st.write("🤖:", result["answer"])





