import os
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter

# スライドバーにアップローダーをセット
uploaded_file = st.sidebar.file_uploader("upload", type="pdf")
# OpenAIのAPIキーをstreamlitの設定から取得
os.environ['OPENAI_API_KEY'] = st.secrets.OpenAIAPI.openai_api_key

# テキスト分割用の設定
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 100,
    length_function = len,
)

# 学習させるpdfファイルを読み込む処理
if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # サイドバーから選択したpdfファイル情報を取得
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # pdfファイルを読み込む
    loader = PyPDFLoader(file_path=tmp_file_path)  
    # ファイルサイズが大きいのでchatgptが分析できるサイズに分割
    data = loader.load_and_split(text_splitter)

    # chatgptで解析するベクトルデータベースを作成
    embeddings = OpenAIEmbeddings()

    # 近似最近傍探索ライブラリを生成
    vectors = FAISS.from_documents(data, embeddings)

    # 回答生成モデルを作成
    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo-16k'),retriever=vectors.as_retriever())

# LLM(chatgpt)による回答生成処理
def conversational_chat(query):

    # 回答生成モデルに質問をセットし回答を出力
    result = chain({"question": query, "chat_history": st.session_state['history']})
    # 質問、回答を履歴にセット
    st.session_state['history'].append((query, result["answer"]))
        
    #　最新履歴データを返す
    return result["answer"]
    
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Feel free to ask about anything regarding this"]
    # st.session_state['generated'] = ["Hello! Feel free to ask about anything regarding this" + uploaded_file.name]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hi!"]
        
# This container will be used to display the chat history.
response_container = st.container()
# This container will be used to display the user's input and the response from the ChatOpenAI model.
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):

        # テキストボックスに入力を促すメッセージを出力
        user_input = st.text_input("Input:", placeholder="ここに質問を入力してください。", key='input')
        # 送信ボタン
        submit_button = st.form_submit_button(label='送信')
            
    if submit_button and user_input:
        # 質問内容からchatgptの回答を取得
        output = conversational_chat(user_input)

        # チャット上に質問内容をセット
        st.session_state['past'].append(user_input)
        # チャット上に回答内容をセット
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            # GUI上に質問者の質問内容を表示
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            # GUI上に回答内容を表示
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
