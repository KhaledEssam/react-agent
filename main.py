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


provider_list = ["ollama", "groq"]
provider_str_to_enum = {
    "ollama": provider.ProviderType.OLLAMA,
    "groq": provider.ProviderType.GROQ,
}
provider_to_model_names = {
    "ollama": ["phi4-mini:3.8b", "command-r7b:7b", "granite3.2:8b", "llama3.1:8b"],
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen-qwq-32b",
        "qwen-2.5-32b",
    ],
}


def main():
    load_dotenv()
    system_prompt = read("prompts/system.txt")
    available_tools = read("prompts/tools.txt")
    system_prompt = system_prompt.replace("{AVAILABLE_TOOLS}", available_tools)

    st.sidebar.title("Settings")
    provider_type = st.sidebar.selectbox("Provider", provider_list)
    provider_enum = provider_str_to_enum[provider_type]
    model_names = provider_to_model_names[provider_type]
    model_name = st.sidebar.selectbox("Model", model_names)

    if prompt := st.chat_input("Ask me a question!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.text(prompt)

        with st.chat_message("assistant"):
            response = agent_loop(
                max_iterations=10,
                system_prompt=system_prompt,
                provider=provider_enum,
                model_name=model_name,
                query=prompt,
            )
            st.text(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
