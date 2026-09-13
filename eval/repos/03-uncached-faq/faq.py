import anthropic
client = anthropic.Anthropic()

POLICY_DOC = "Refund policy: " + ("All refunds are processed within 14 days subject to review. " * 400)

def answer_faq(question: str) -> str:
    """Sends the full 400-paragraph policy doc on every single question, uncached."""
    msg = client.messages.create(
        model="claude-sonnet-4",
        max_tokens=400,
        system=POLICY_DOC,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text
