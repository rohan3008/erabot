import openai
client = openai.OpenAI()

def run_until_done(task: str) -> str:
    history = [{"role": "user", "content": task}]
    while True:
        resp = client.chat.completions.create(model="gpt-4o", messages=history)
        out = resp.choices[0].message.content
        history.append({"role": "assistant", "content": out})
        if "TASK_COMPLETE" in out:
            return out
