import openai
client = openai.OpenAI()

STYLE_GUIDE = "Write concisely. " * 500

def summarize_ticket(ticket: str) -> str:
    """gpt-4o for one-line summaries, with a 500x-duplicated style guide."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": STYLE_GUIDE + STYLE_GUIDE},
            {"role": "user", "content": "One line summary: " + ticket},
        ],
    )
    return resp.choices[0].message.content
