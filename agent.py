import re
import yaml

import provider
import tools


class Agent:
    def __init__(
        self,
        provider_type: provider.ProviderType,
        model_name: str,
        system_prompt: str = "",
    ):
        self.provider_type = provider_type
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages = []

        match self.provider_type:
            case provider.ProviderType.OLLAMA:
                self.client = provider.OllamaProvider(model_name)
            case provider.ProviderType.GROQ:
                self.client = provider.GroqProvider(model_name)

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def __call__(self, message: str = ""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion: str = self.client.chat(
            messages=self.messages,
            temperature=0,
            stop=["<end>"],
        )
        return completion


def agent_loop(
    max_iterations: int, provider: str, model_name: str, system_prompt: str, query: str
):
    agent = Agent(provider, model_name, system_prompt)
    next_prompt = query

    output = []

    for i in range(max_iterations):
        output.append(f"Iteration {i}")
        result = agent(next_prompt)
        output.append(result)

        if "Answer" in result:
            break

        action = re.search(r"Action:(.*)PAUSE", result, re.MULTILINE | re.DOTALL)

        if not action:
            continue

        regex_match = action.group(1).strip()

        try:
            tool_call = yaml.load(regex_match, Loader=yaml.FullLoader)
            chosen_tool = tool_call["name"]
        except yaml.YAMLError as exc:
            observation = f"Error parsing tool call: {exc}"
            next_prompt = f"Observation: {observation}"
            output.append(f"Next prompt: {next_prompt}")
            continue

        valid = True

        match chosen_tool:
            case "add":
                fn = tools.add
            case "divide":
                fn = tools.divide
            case "multiply":
                fn = tools.multiply
            case "subtract":
                fn = tools.subtract
            case "search":
                fn = tools.search
            case _:
                valid = False

        if valid:
            observation = fn(**tool_call["args"])
        else:
            observation = "Invalid tool invokation"
        next_prompt = f"Observation: {observation}"
        output.append(f"Next prompt: {next_prompt}")
    return "\n".join(output)
