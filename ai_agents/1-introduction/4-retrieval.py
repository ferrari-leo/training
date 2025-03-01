import os
import json
import requests
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# model = "llama3.2:1b"
model = "mistral"
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Define knowledge base retrieval tool


def search_kb(question: str):
    """
    Load knowledge base from json file (mock function for demo purposes)
    """
    with open("./kb.json", "r") as f:
        return json.load(f)


# Call model with search_kb tool defined

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get answer to user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(model=model, messages=messages, tools=tools)

# Check model outputs

completion.model_dump()

# Execute search_kb function


def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

# supply result and call model again


class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    source: int = Field(description="The record id of the answer")


completion2 = client.beta.chat.completions.parse(
    model=model, messages=messages, tools=tools, response_format=KBResponse
)

# check model response

final_response = completion2.choices[0].message.parsed

# Check question that doesn't trigger the model

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather in Tokyo?"},
]

completion3 = client.beta.chat.completions.parse(
    model=model, messages=messages, tools=tools
)

completion3.choices[0].message.content
completion3.model_dump()
