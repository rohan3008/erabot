import anthropic
client = anthropic.Anthropic()

def handle(query: str) -> dict:
    summary = client.messages.create(
        model="claude-sonnet-4", max_tokens=300,
        messages=[{"role": "user", "content": "Summarize: " + query}],
    ).content[0].text
    # identical call repeated to build the title — same model, same input
    title = client.messages.create(
        model="claude-sonnet-4", max_tokens=300,
        messages=[{"role": "user", "content": "Summarize: " + query}],
    ).content[0].text
    return {"title": title[:60], "summary": summary}
