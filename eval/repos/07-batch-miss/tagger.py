import openai
client = openai.OpenAI()

def tag_products(descriptions: list[str]) -> list[str]:
    """Tags 500+ products one call at a time, sequentially."""
    tags = []
    for desc in descriptions:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "One-word category for: " + desc}],
        )
        tags.append(resp.choices[0].message.content)
    return tags
