import openai
client = openai.OpenAI()

def judge_answer(question: str, answer: str) -> str:
    """Offline eval judge — runs nightly over 10k pairs on gpt-4-turbo."""
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": f"Q: {question}\nA: {answer}\nGrade PASS or FAIL."}],
    )
    return resp.choices[0].message.content
