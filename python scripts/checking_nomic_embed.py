import math
import requests

OLLAMA_EMBED_URL = "http://localhost:11436/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
TIMEOUT_SEC = 60
EXPECTED_DIM = 768

def get_embedding(text: str):
    payload = {"model": EMBED_MODEL, "prompt": text}
    r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError(f"No embedding in response: {data}")
    if len(emb) != EXPECTED_DIM:
        raise RuntimeError(f"Bad embedding dim={len(emb)} expected={EXPECTED_DIM}")
    return emb

def cosine(a, b) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-12
    nb = math.sqrt(sum(y*y for y in b)) + 1e-12
    return dot / (na * nb)

def main():
    print(f"Embedding model: {EMBED_MODEL}")
    print("Enter prompt 1 (finish with Enter):")
    p1 = input("> ").strip()

    print("\nEnter prompt 2 (finish with Enter):")
    p2 = input("> ").strip()

    if not p1 or not p2:
        print("Both prompts must be non-empty.")
        return

    e1 = get_embedding(p1)
    e2 = get_embedding(p2)

    sim = cosine(e1, e2)
    print(f"\nCosine similarity = {sim:.6f}")

    # Optional: quick interpretation
    if sim >= 0.80:
        msg = "Very similar intent / near-duplicate"
    elif sim >= 0.60:
        msg = "Same general task family"
    elif sim >= 0.40:
        msg = "Loosely related"
    else:
        msg = "Mostly different"
    print(f"Interpretation: {msg}")

if __name__ == "__main__":
    main()
