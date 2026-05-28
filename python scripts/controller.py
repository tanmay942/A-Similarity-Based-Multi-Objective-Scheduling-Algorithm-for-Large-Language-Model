import requests
import json
import time
import os
from typing import Optional, Tuple, Dict, Any

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11436/api/generate"

# Run one model fully, then next (RAM-friendly)
MODELS = [
    "llama3",
    "qwen2.5:3b-instruct",
    "mistral",
]

PROMPTS_FILE = "kb_5000.json"   # list of {"id": ..., "prompt": ...}
OUT_DIR = "ollama_results"

TEMPERATURE = 0.7

# IMPORTANT:
# For "true model behavior", keep this as None (no artificial output cap).
# If you NEED a cap for runtime reasons, set e.g. 256 / 512.
NUM_PREDICT: Optional[int] = None

TIMEOUT_SEC = 900           # per request timeout
RETRIES = 2                 # retry count on transient errors
SLEEP_BETWEEN = 0.5         # small delay
CHECKPOINT_EVERY = 5       # save after every N prompts

# For research you said: "no prompt length limit"
# So keep it None. If laptop struggles, set e.g. 8000.
MAX_PROMPT_CHARS: Optional[int] = None


# =========================
# HELPERS
# =========================
def safe_prompt(text: str) -> str:
    """Optional prompt truncation (disabled by default)."""
    if MAX_PROMPT_CHARS is None:
        return text
    if len(text) <= MAX_PROMPT_CHARS:
        return text
    return text[:MAX_PROMPT_CHARS] + "\n\n[TRUNCATED_FOR_LENGTH]"


def atomic_save_json(path: str, obj: Any) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_ollama(prompt_text: str, model_name: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Returns: (response_text_or_None, meta_dict)
    meta includes: latency_ms, input_tokens, output_tokens, total_tokens, done_reason, done, error.
    """
    options = {"temperature": TEMPERATURE}

    # Only include num_predict if the user set it (otherwise no artificial cap)
    if NUM_PREDICT is not None:
        options["num_predict"] = NUM_PREDICT

    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": options
    }

    last_error = None
    start = None
    for attempt in range(RETRIES + 1):
        start = time.time()
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
        except requests.exceptions.RequestException as e:
            last_error = f"REQUEST_EXCEPTION: {str(e)}"
            # small backoff
            time.sleep(0.5 * (attempt + 1))
            continue

        latency_ms = int((time.time() - start) * 1000)

        if r.status_code != 200:
            last_error = f"HTTP_{r.status_code}: {r.text}"
            time.sleep(0.5 * (attempt + 1))
            continue

        try:
            data = r.json()
        except Exception as e:
            last_error = f"BAD_JSON: {str(e)} | raw={r.text[:500]}"
            time.sleep(0.5 * (attempt + 1))
            continue

        response_text = data.get("response", "")

        prompt_tokens = data.get("prompt_eval_count", None)
        output_tokens = data.get("eval_count", None)
        done_reason = data.get("done_reason", None)
        done_flag = data.get("done", None)

        meta = {
            "latency_ms": latency_ms,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (
                (prompt_tokens or 0) + (output_tokens or 0)
                if (prompt_tokens is not None or output_tokens is not None)
                else None
            ),
            "done": done_flag,
            "done_reason": done_reason,
            "error": None
        }

        # If output cap is set and model hit it, record it clearly
        if NUM_PREDICT is not None and output_tokens == NUM_PREDICT:
            meta["hit_num_predict_cap"] = True
        else:
            meta["hit_num_predict_cap"] = False

        return response_text, meta

    # all attempts failed
    latency_ms = int((time.time() - start) * 1000) if start is not None else None
    return None, {"latency_ms": latency_ms, "error": last_error}


def wrap_prompt(user_prompt: str) -> str:
    return (
        "You are an AI assistant.\n\n"
        "User prompt:\n"
        f"{user_prompt}\n\n"
        "Provide the best possible answer."
    )


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = load_json(PROMPTS_FILE)  # expects list of {"id":..., "prompt":...}

    # Basic validation
    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError("PROMPTS_FILE must be a non-empty list of prompts.")

    for model_name in MODELS:
        print(f"\n🚀 Starting model: {model_name}")
        result_file = os.path.join(OUT_DIR, f"results_{model_name}.json")

        # Resume support
        results = []
        completed_ids = set()

        if os.path.exists(result_file):
            try:
                results = load_json(result_file)
                completed_ids = {r["prompt_id"] for r in results if "prompt_id" in r}
                print(f"🔁 Resuming {model_name}: completed {len(completed_ids)} prompts")
            except Exception:
                print("⚠️ Previous results unreadable. Starting fresh.")
                results, completed_ids = [], set()

        processed_since_save = 0

        for task in tasks:
            pid = task.get("id")
            if pid is None:
                # If your file uses a different key, adjust here
                continue
            if pid in completed_ids:
                continue

            user_prompt = safe_prompt(task.get("prompt", ""))

            # IMPORTANT: Keep prompt unchanged other than the wrapper
            wrapped = wrap_prompt(user_prompt)

            print(f"▶ Prompt {pid} | Model {model_name}")

            response_text, meta = call_ollama(wrapped, model_name)

            record = {
                "prompt_id": pid,
                "model": model_name,
                "prompt": user_prompt,       # store prompt for reproducibility
                "response": response_text,
                **meta
            }

            results.append(record)
            completed_ids.add(pid)

            processed_since_save += 1
            if processed_since_save >= CHECKPOINT_EVERY:
                atomic_save_json(result_file, results)
                print(f"💾 Checkpoint saved: {result_file}")
                processed_since_save = 0

            time.sleep(SLEEP_BETWEEN)

        atomic_save_json(result_file, results)
        print(f"✅ Finished model: {model_name}")
        print(f"💾 Saved: {result_file}")
        print(f"🧹 Optional: ollama stop {model_name}\n")

    print("\n✅ All models done.")


if __name__ == "__main__":
    main()
