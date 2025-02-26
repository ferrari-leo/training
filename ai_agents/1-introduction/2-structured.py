from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Define response format in Pydantic Model


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# Call model

completion = client.beta.chat.completions.parse(
    model="llama3.2:1b",
    messages=[
        {
            "role": "system",
            "content": "Extract the event information.",
        },
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    response_format=CalendarEvent,
)

# Parse response

event = completion.choices[0].message.parsed
print(event)
