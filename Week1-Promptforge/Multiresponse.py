from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY1"))

# conversation history — you manage this list manually
conversation = [
    {
        "role": "system",
        "content": "You are a concise technical explainer. Keep responses under 80 words."
    }
]

def chat(user_message: str) -> str:
    conversation.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation,
        temperature=0.7,
        max_tokens=200,
    )

    reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})
    return reply

print(chat("What is a transformer model?"))
print(chat("How does the attention mechanism work in it?"))  # model remembers context