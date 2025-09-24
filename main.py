import os

from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import chat_models
from langchain_perplexity import ChatPerplexity
from langchain_community.chat_models import ChatOllama
from langchain.globals import set_debug
set_debug(True)
import streamlit as st

PERPLEXITY_API_KEY = os.getenv("OPENAI_API_KEY")
# PERPLEXITY_API_KEY = os.getenv("PAK")
# print(PERPLEXITY_API_KEY)
# print(OPENAI_API_KEY)

# llm = ChatPerplexity(temperature=0, api_key=PERPLEXITY_API_KEY, model="sonar")
# st.title("Capital's info")
llm = ChatOllama(model="llama3.2")


title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    you need to craft an impactful title for a speech
    on the following topic: {topic}
    Answer Exactly with one title"""
)

speech_prompt = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""You are an experienced speech writer.
    You need to write a powerful {emotion} speech of 350 words for the following title: {title}
    Format the output with 2 keys: Title: and Speech: and fill them with there respective value"""
)

first_chain = title_prompt | llm | StrOutputParser()
second_chain = speech_prompt | llm
final_chain = first_chain | (lambda title:{"title":title, "emotion":emotion} )|second_chain

st.title("Speech Generator")
topic = st.text_input("Enter the topic:")
emotion = st.text_input("Enter the emotion:")

if topic and emotion:
    response = final_chain.invoke({"topic": topic})
    st.write(response.content)



from openai import api_key

# PERPLEXITY_API_KEY = os.getenv("OPENAI_API_KEY")
# # PERPLEXITY_API_KEY = os.getenv("PAK")
# # print(PERPLEXITY_API_KEY)
# # print(OPENAI_API_KEY)
#
# # llm = ChatPerplexity(temperature=0, api_key=PERPLEXITY_API_KEY, model="sonar")
# st.title("Speech Generator")
# llm = ChatOllama(model="llama3.2")
# question = st.text_input("Enter the country:")
# no_of_para = st.number_input("Enter the paragraphs:", min_value=1, max_value=5)
# language = st.text_input("Enter the language")
#
# chain = prompt_template | llm
#
# if question:
#     response = chain.invoke({"country":question,
#                              "no_of_para":no_of_para,
#                              "language":language})
#     st.write(response.content)
