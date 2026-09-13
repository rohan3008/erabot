import openai, glob
client = openai.OpenAI()

def build_index():
    """Called at every startup — re-embeds EVERY document, changed or not."""
    vectors = {}
    for path in glob.glob("docs/**/*.md", recursive=True):
        text = open(path).read()
        emb = client.embeddings.create(model="text-embedding-3-large", input=text)
        vectors[path] = emb.data[0].embedding
    return vectors
