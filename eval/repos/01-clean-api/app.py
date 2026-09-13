from fastapi import FastAPI
app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "price": item_id * 2.5}

@app.get("/health")
def health():
    return {"ok": True}
