import openai
client = openai.OpenAI()

def is_spam(text: str) -> bool:
    """Binary spam check — runs on gpt-4o for a yes/no answer."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Reply only 'spam' or 'ham'."},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip() == "spam"
