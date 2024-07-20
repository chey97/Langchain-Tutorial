from langchain_community.agent_toolkits.openapi import planner
from langchain_openai import ChatOpenAI
from langchain_community.utilities import TextRequestsWrapper
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
import os
import yaml

ALLOW_DANGEROUS_REQUEST = True

with open("path_to_yaml_file") as f:
    raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)
openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
# print(openai_api_spec)  

# headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
# openai_requests_wrapper = TextRequestsWrapper(headers=headers)
openai_requests_wrapper = TextRequestsWrapper()

llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0.25
    )

openai_agent = planner.create_openapi_agent(
    openai_api_spec, 
    openai_requests_wrapper, 
    llm,
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
)

user_query = "What is the sensor temperature at 1696111203193, which sensor"
openai_agent.invoke(user_query)

