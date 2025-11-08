from typing_extensions import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

client = OpenAI()

class State(TypedDict):
    user_query: str
    llm_result: str | None

def general_query(state: State) -> State:
    query = state["user_query"]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": query}]
    )
    state["llm_result"] = response.choices[0].message.content
    return state

graph_builder = StateGraph(State)
graph_builder.add_node("general_query", general_query)
graph_builder.add_edge(START, "general_query")
graph_builder.add_edge("general_query", END)

graph = graph_builder.compile()

def main():
    user = input("Query: ")
    state: State = {"user_query" : user, "llm_result": None}
    for event in graph.stream(state):
        print(event)

main()
