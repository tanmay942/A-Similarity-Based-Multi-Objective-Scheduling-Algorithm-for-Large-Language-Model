import json
import time
from typing import Any, Dict, List, Tuple

import requests

# =========================
# CONFIG
# =========================
TEST_PROMPTS_FILE = "test_500.json"   # or your test file
KB_EMB_FILE = "prompt_embeddings.json"               # embeddings for the 100 KB prompts

OLLAMA_EMB_URL = "http://localhost:11436/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

K = 5
TIMEOUT_SEC = 60
SLEEP_BETWEEN = 0.02
EXPECTED_DIM = 768


# =========================
# HELPERS
# =========================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    r = requests.post(OLLAMA_EMB_URL, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError(f"No embedding in response: {data}")
    if len(emb) != EXPECTED_DIM:
        raise RuntimeError(f"Bad embedding dim={len(emb)} expected={EXPECTED_DIM}")
    return emb


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: List[float]) -> float:
    return (sum(x * x for x in a) ** 0.5) + 1e-12


def cosine(a: List[float], b: List[float]) -> float:
    return dot(a, b) / (norm(a) * norm(b))


def topk_similar(query_emb: List[float], kb: List[Dict[str, Any]], k: int) -> List[Tuple[int, float]]:
    sims: List[Tuple[int, float]] = []
    for row in kb:
        pid = row.get("id")
        emb = row.get("embedding")
        if not isinstance(pid, int) or not isinstance(emb, list) or len(emb) != EXPECTED_DIM:
            continue
        sims.append((pid, cosine(query_emb, emb)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


# =========================
# MAIN
# =========================
def main():
    kb = load_json(KB_EMB_FILE)          # [{ "id": 1, "embedding": [...] }, ...]
    tests = load_json(TEST_PROMPTS_FILE) # [{ "prompt_id":..., "prompt":"..." }, ...]

    if not isinstance(kb, list) or not kb:
        raise ValueError("KB_EMB_FILE must be a non-empty list")
    if not isinstance(tests, list) or not tests:
        raise ValueError("TEST_PROMPTS_FILE must be a non-empty list")

    print(f"Loaded KB embeddings: {len(kb)}")
    print(f"Loaded test prompts:  {len(tests)}")
    print(f"Using k={K} and model={EMBED_MODEL}\n")

    for idx, item in enumerate(tests, start=1):
        prompt = item.get("prompt", "")
        pid = item.get("prompt_id", item.get("id", idx))

        if not isinstance(prompt, str) or not prompt.strip():
            continue

        print(f"=== Test #{idx} | prompt_id={pid} ===")
        print(prompt[:180].replace("\n", " ") + ("..." if len(prompt) > 180 else ""))
        qemb = get_embedding(prompt)

        nbrs = topk_similar(qemb, kb, K)

        print("Top-5 similar KB prompts (id, cosine):")
        for kb_id, sim in nbrs:
            print(f"  - {kb_id:>4}   {sim:.4f}")
        print()

        time.sleep(SLEEP_BETWEEN)


if __name__ == "__main__":
    main()
