from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.bigram_model import BigramModel
import spacy
from functools import lru_cache
from fastapi import HTTPException

app = FastAPI(title="Text Generation and Word Embeddings")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned "
    "and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1)
    length: int = Field(ge=1, le=100)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    text = bigram_model.generate_text(
        request.start_word, request.length
    )
    return {"generated_text": text}


@lru_cache(maxsize=1)
def load_nlp():
    return spacy.load("en_core_web_md")


@app.get("/embedding")
def get_embedding(word: str):
    word = word.strip().lower()
    if not word or len(word.split()) != 1:
        raise HTTPException(status_code=400, detail="Enter one word.")
    token = load_nlp().vocab[word]
    if not token.has_vector:
        raise HTTPException(status_code=404, detail="Word not found.")
    return {"word": word, "embedding": token.vector.tolist()}

