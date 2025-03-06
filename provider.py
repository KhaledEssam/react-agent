import ollama
import groq
from enum import Enum


class Provider:
    def __init__(self):
        pass

    def chat(self, messages: list, stop: list, temperature: float):
        pass


class OllamaProvider(Provider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client()

    def chat(self, messages: list, stop: list, temperature: float = 0):
        completion: ollama.ChatResponse = self.client.chat(
            messages=messages,
            model=self.model_name,
            options={"temperature": temperature, "stop": stop},
        )
        return completion.message.content


class GroqProvider(Provider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = groq.Groq()

    def chat(self, messages: list, stop: list, temperature: float = 0):
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=temperature,
            stop=stop,
            service_tier="auto",
        )
        return completion.choices[0].message.content


class ProviderType(Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
