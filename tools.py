from langchain_community.tools import DuckDuckGoSearchRun


def add(arg1: float, arg2: float) -> float:
    return arg1 + arg2


def divide(arg1: float, arg2: float) -> float:
    return arg1 / arg2


def subtract(arg1: float, arg2: float) -> float:
    return arg1 - arg2


def multiply(arg1: float, arg2: float) -> float:
    return arg1 * arg2


search_tool = DuckDuckGoSearchRun()


def search(query: str) -> str:
    print(f"========== Searching for: {query} ==========")
    return search_tool.run(query)
