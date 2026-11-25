import streamlit as st

st.set_page_config(
    page_title="Локальный MLOps-помощник",
    page_icon="🤖",
)

st.title("Локальный MLOps-помощник")
st.caption("Учебный чат с моделью, работающей на этом компьютере.")

# История сохраняется между повторными выполнениями страницы.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Показываем предыдущие сообщения.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Принимаем новое сообщение.
if prompt := st.chat_input("Введите вопрос"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)