import os

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def generate_answer(question, context):

    prompt = f"""
Answer ONLY from the provided context.

If answer is not available, say:
"The documents do not contain enough information."

Answer in the same language as the question.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content