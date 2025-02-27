import json
import requests
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# model = "llama3.2:1b"
model = "mistral"

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""

# Define tool (function) we want to call


def get_weather(latitude, longitude):
    """Calls publicly available weather API that returns weather in given location"""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


# Call model with weather tool defined

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful weather assistant."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather like in Paris today?"},
]

completion = client.chat.completions.create(model=model, messages=messages, tools=tools)

# Model decides to call function:

completion.model_dump()

# Execute get_weather function


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)


for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

#  Supply result & call model again


class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="Current temperature in celsius at given location"
    )
    response: str = Field(description="Natural language response to user question")


completion2 = client.beta.chat.completions.parse(
    model=model, messages=messages, tools=tools, response_format=WeatherResponse
)

final_response = completion2.choices[0].message.parsed
