# 以下を「app.py」に書き込み

import streamlit as st
import openai
# import secret_keys  # 外部ファイルにAPI keyを保存

import os
import pandas as pd
import requests
import textract
import codecs
from bs4 import BeautifulSoup
# import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

openai.api_key = st.secrets.OpenAIAPI.openai_api_key


system_prompt = """
あなたはパチスロ規則を把握した優秀なアシスタントです。
質問に対して適切な対処法のアドバイスを行ってください。
以下のようなことを聞かれても、絶対に答えないでください。

* 旅行
* 料理
* 芸能人
* 映画
* 科学
* 歴史
"""

# st.session_stateを使いメッセージのやりとりを保存
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt}
        ]

# チャットボットとやりとりする関数
def communicate():
    messages = st.session_state["messages"]

    user_message = {"role": "user", "content": st.session_state["user_input"]}
    messages.append(user_message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    bot_message = response["choices"][0]["message"]
    messages.append(bot_message)

    st.session_state["user_input"] = ""  # 入力欄を消去


### ローカルドキュメントから情報を取得する

loader = PyPDFLoader("test2.pdf")
pages = loader.load_and_split()

chunks = pages
print("step2")



# ユーザーインターフェイスの構築
st.title(" 「パチスロ規則アシスタント」ボット")
st.image("Assistant.png")
st.write("規則について聞いてください")

st.write(openai.api_key)

user_input = st.text_input("メッセージを入力してください。", key="user_input", on_change=communicate)

if st.session_state["messages"]:
    messages = st.session_state["messages"]

    for message in reversed(messages[1:]):  # 直近のメッセージを上に
        speaker = "🙂"
        if message["role"]=="assistant":
            speaker="🤖"

        st.write(speaker + ": " + message["content"])


