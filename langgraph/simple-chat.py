from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def main():
    user = input("Query: ")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": user}]
    )

    print(response.choices[0].message.content)

main()
