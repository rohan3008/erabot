import functools
import openai
client = openai.OpenAI()

@functools.lru_cache(maxsize=4096)
def classify(text: str) -> str:
    """Cheap model, tiny prompt, cached, bounded — nothing to fix here."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=4,
        messages=[{"role": "user", "content": "positive or negative: " + text[:280]}],
    )
    return resp.choices[0].message.content.strip()
