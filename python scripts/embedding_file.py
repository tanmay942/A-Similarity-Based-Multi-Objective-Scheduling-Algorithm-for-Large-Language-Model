import json
import os
import time
import requests
from typing import List, Dict, Any, Optional

# =========================
# CONFIG
# =========================
OLLAMA_EMBED_URL = "http://localhost:11436/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

INPUT_FILE = "kb_5000.json"     # [{ "id": 1, "prompt": "..."}, ...]
OUT_FILE = "prompt_embeddings.json"          # [{ "id": 1, "embedding": [...]}, ...]
ERROR_FILE = "prompt_embeddings_errors.json" # [{ "id": 1, "error": "..."} , ...]
CHECKPOINT_EVERY = 10
SLEEP_BETWEEN = 0.05
TIMEOUT_SEC = 180

# =========================
# HELPERS
# =========================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_atomic(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def get_embedding(text: str) -> Optional[List[float]]:
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP_{r.status_code}: {r.text}")
    data = r.json()
    emb = data.get("embedding", None)
    if emb is None:
        raise RuntimeError(f"No 'embedding' in response: {data}")
    return emb

# =========================
# MAIN
# =========================
def main():
    # Load tasks
    tasks = load_json(INPUT_FILE)

    # Resume support
    embeddings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    done_ids = set()

    if os.path.exists(OUT_FILE):
        try:
            embeddings = load_json(OUT_FILE)
            done_ids = {x["id"] for x in embeddings if "id" in x}
            print(f"🔁 Resuming: already embedded {len(done_ids)} prompts.")
        except Exception:
            print("⚠️ Could not read existing output file. Starting fresh.")
            embeddings = []
            done_ids = set()

    if os.path.exists(ERROR_FILE):
        try:
            errors = load_json(ERROR_FILE)
        except Exception:
            errors = []

    processed_since_save = 0

    for item in tasks:
        pid = item.get("id")
        prompt = item.get("prompt", "")

        if pid is None:
            continue

        if pid in done_ids:
            continue

        try:
            emb = get_embedding(prompt)
            embeddings.append({
                "id": pid,
                "embedding": emb
            })
            done_ids.add(pid)

            print(f"✅ Embedded id={pid} | dim={len(emb)}")

        except Exception as e:
            err_msg = str(e)
            print(f"❌ Error id={pid}: {err_msg}")
            errors.append({
                "id": pid,
                "error": err_msg
            })

        processed_since_save += 1

        # checkpointing
        if processed_since_save >= CHECKPOINT_EVERY:
            save_json_atomic(OUT_FILE, embeddings)
            save_json_atomic(ERROR_FILE, errors)
            print(f"💾 Checkpoint saved ({len(embeddings)} embeddings)")
            processed_since_save = 0

        time.sleep(SLEEP_BETWEEN)

    # final save
    save_json_atomic(OUT_FILE, embeddings)
    save_json_atomic(ERROR_FILE, errors)

    print("\n✅ DONE")
    print(f"📦 Embeddings saved to: {OUT_FILE}")
    print(f"⚠️ Errors saved to: {ERROR_FILE}")
    print(f"Total embedded: {len(embeddings)} / {len(tasks)}")

if __name__ == "__main__":
    main()
