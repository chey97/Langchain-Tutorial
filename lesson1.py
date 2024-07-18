from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    )

# response = llm.invoke("Write a poem about AI")
# print(response)

# response = llm.batch(["Write a poem about AI","Hello, how are you?"])
# print(response)

response = llm.stream("Write a poem about AI")
# print(response)

for chunk in response:
    print(chunk.content, end="")