from agent import agent_loop
from utils import read
from dotenv import load_dotenv
import provider


def main():
    load_dotenv()
    system_prompt = read("prompts/system.txt")
    available_tools = read("prompts/tools.txt")
    system_prompt = system_prompt.replace("{AVAILABLE_TOOLS}", available_tools)

    agent_loop(
        max_iterations=10,
        system_prompt=system_prompt,
        provider=provider.ProviderType.GROQ,
        model_name="qwen-qwq-32b",
        query="I bought an nvidia stock at $80. will I make a proffit or not if I sell it now?",
    )


if __name__ == "__main__":
    main()
