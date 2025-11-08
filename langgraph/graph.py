from typing_extensions import TypedDict
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

load_dotenv()
client = OpenAI()

class ClassifyMessageResponse(BaseModel):
    is_coding_question: bool

class CodeAccuracyResponse(BaseModel):
    accuracy_percentage: str

class State(TypedDict):
    user_query: str
    llm_result: str | None
    is_coding_question: bool | None
    accuracy_percentage: str | None


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



def route_query(state: State) -> Literal["general_query", "coding_query"]:
    return "coding_query" if state["is_coding_question"] else "general_query"

def general_query(state: State) -> State:
    query = state["user_query"]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": query}]
    )
    state["llm_result"] = response.choices[0].message.content
    return state

def coding_query(state: State) -> State:
    query = state["user_query"]
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": query}]
    )
    state["llm_result"] = response.choices[0].message.content
    return state

def coding_validation(state: State) -> State:
    query = state["user_query"]
    llm_code_or_answer = state["llm_result"]

    system = (
        "You are an expert in evaluating code accuracy"
        "Return the percentage of accuracy in json {accuracy_percentage: str}"
        f"User query: {query}"
        f"Code: {llm_code_or_answer}"
    )
    response = client.chat.completions.parse(
        model = "gpt-4.1",
        response_format=CodeAccuracyResponse,
        messages = [
            {"role":"system", "content": system},
            {"role": "user", "content": query},
        ],
    )
    state["accuracy_percentage"] = response.choices[0].message.parsed.accuracy_percentage
    return state

graph_builder = StateGraph(State)
graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("general_query", general_query)
graph_builder.add_node("coding_query", coding_query)
graph_builder.add_node("coding_validation", coding_validation)


graph_builder.add_edge(START,"classify_message")
graph_builder.add_conditional_edges("classify_message", route_query)
graph_builder.add_edge("general_query",END)
graph_builder.add_edge("coding_query","coding_validation")
graph_builder.add_edge("coding_validation",END)
graph = graph_builder.compile()

def main():
    user = input("Query: ")
    state: State = {"user_query": user, "llm_result": None, "is_coding_question": None}
    for event in graph.stream(state):
        print(event)

main()
