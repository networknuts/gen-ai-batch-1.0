from typing_extensions import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

load_dotenv()
client = OpenAI()

class ClassifyMessageResponse(BaseModel):
    is_coding_question: bool

class State(TypedDict):
    user_query: str
    llm_result: str | None
    is_coding_question: bool | None


def classify_message(state: State) -> State:
    query = state["user_query"]
    system = ("You are an AI assistant. Detect if the user query is a coding question. Return JSON: {is_coding_question: boolean}")

    response = client.chat.completions.parse(
        model="gpt-4.1-mini",
        response_format=ClassifyMessageResponse,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )
    state["is_coding_question"] = response.choices[0].message.parsed.is_coding_question
    return state

def general_query(state: State) -> State:
    query = state["user_query"]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": query}]
    )
    state["llm_result"] = response.choices[0].message.content
    return state


graph_builder = StateGraph(State)
graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("general_query", general_query)
graph_builder.add_edge(START,"classify_message")
graph_builder.add_edge("classify_message","general_query")
graph_builder.add_edge("general_query",END)
graph = graph_builder.compile()

def main():
    user = input("Query: ")
    state: State = {"user_query": user, "llm_result": None, "is_coding_question": None}
    for event in graph.stream(state):
        print(event)

main()
