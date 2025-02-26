from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

completion = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[
        {"role": "system", "content": "You're a helpful assistant"},
        {
            "role": "user",
            "content": "Write a Gen Z appraisal of AI",
        },
    ],
)

response = completion.choices[0].message.content
print(response)
