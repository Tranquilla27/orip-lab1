import streamlit as st
from openai import APIError, OpenAI

st.set_page_config(
    page_title="Локальный MLOps-помощник",
    page_icon="🤖",
)

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    timeout=120.0,
    max_retries=0,
)

st.title("Локальный MLOps-помощник")
st.caption("Модель mlops-bot работает на этом компьютере.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Очистить диалог"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Введите вопрос"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with client.chat.completions.create(
                model="mlops-bot",
                messages=st.session_state.messages,
                temperature=0.3,
                max_tokens=512,
                stream=True,
            ) as stream:
                response_text = st.write_stream(
                    stream,
                    cursor="▌",
                )
        except APIError as error:
            st.error(f"Не удалось получить ответ от Ollama: {error}")
            st.session_state.messages.pop()
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
            })