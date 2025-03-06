from agent import agent_loop
from utils import read
from dotenv import load_dotenv
import provider


import streamlit as st


st.title("QA Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def main():
    load_dotenv()
    system_prompt = read("prompts/system.txt")
    available_tools = read("prompts/tools.txt")
    system_prompt = system_prompt.replace("{AVAILABLE_TOOLS}", available_tools)

    if prompt := st.chat_input("Ask me a question!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.text(prompt)

        with st.chat_message("assistant"):
            response = agent_loop(
                max_iterations=10,
                system_prompt=system_prompt,
                provider=provider.ProviderType.GROQ,
                model_name="qwen-qwq-32b",
                query=prompt,
            )
            st.text(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
