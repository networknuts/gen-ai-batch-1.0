from dotenv import load_dotenv
from openai import OpenAI
from typing_extensions import TypedDict

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
    print(state["llm_result"])

def main():
    user = input("Query: ")
    state: State = {"user_query": user, "llm_result": None}
    general_query(state)
    print("Final State:", state)

main()
