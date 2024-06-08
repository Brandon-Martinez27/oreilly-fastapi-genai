import requests
import streamlit as st

st.title("FastAPI ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        st.markdown(content)

if prompt := st.chat_input("Write your prompt in this input field"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.text(prompt)

    with st.chat_message("assistant"):
        response = requests.get(
            f"http://localhost:8000/generate/text?prompt={prompt}"
        ).text
        st.text(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": "Here is your generated image"}
    )