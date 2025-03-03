from agent import agent_loop
from utils import read
from dotenv import load_dotenv


def main():
    load_dotenv()
    system_prompt = read("prompts/system.txt")
    available_tools = read("prompts/tools.txt")
    system_prompt = system_prompt.replace("{AVAILABLE_TOOLS}", available_tools)

    agent_loop(
        max_iterations=10,
        system_prompt=system_prompt,
        model_name="qwen-2.5-32b",
        query="what's 2 * the age of Jeff Bezos?",
    )


if __name__ == "__main__":
    main()
