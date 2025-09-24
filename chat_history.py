from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain.globals import set_debug
from langchain_core.runnables.history import RunnableWithMessageHistory
set_debug(True)
import streamlit as st



llm = ChatOllama(model="llama3.2")



template = ChatPromptTemplate.from_messages([
    ("system", "You are an agile coach, Answer any questions"
               "Related to the agile process"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = template | llm
history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id:history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

st.title("Agile Coach AI")

question = st.text_input("Enter the question you want to ask")
if question:
    reesponse = chain_with_history.invoke({"input":question},
                                           {"configurable":{"session_id":"abc123"}})
    st.write(reesponse.content)


st.write("HISTORY")
st.write(history_for_chain)

