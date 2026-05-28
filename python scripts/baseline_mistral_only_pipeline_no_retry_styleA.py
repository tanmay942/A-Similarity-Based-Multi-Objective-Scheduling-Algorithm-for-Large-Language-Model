import os, json, time, random
import requests
from typing import Any, Dict, List, Optional, Tuple

# =========================
# CONFIG (knowledge-free baseline)
# =========================
OLLAMA_GEN_URL = "http://localhost:11436/api/generate"

TEST_PROMPTS_FILE = "test_500.json"   # list of { "prompt_id":..., "prompt":"..." }
OUT_DIR = "online_baseline_runs"

MODELS = ["llama3", "qwen2.5:3b-instruct", "mistral"]

# Generation (match your earlier environment)
GEN_TEMPERATURE = 0.7
GEN_TIMEOUT_SEC = 900   # NO RETRIES (single call like your old baseline scripts)

# Judge (deterministic)
JUDGE_MODEL = "llama3"
JUDGE_TIMEOUT_SEC = 900

SLEEP_BETWEEN = 0.05
CHECKPOINT_EVERY = 5

# For random baseline reproducibility
RANDOM_SEED = 12345

JUDGE_SINGLE_PROMPT = "You are an impartial evaluator.\n\nGiven a USER PROMPT and a single ASSISTANT ANSWER, score the answer from 0 to 10 using:\n- Instruction following (0-4)\n- Correctness / factuality (0-4)\n- Clarity & completeness (0-2)\n\nRules:\n- Penalize hallucinations and confident wrong statements heavily.\n- If the answer is unsafe, refuses incorrectly, is off-topic, or empty, score low.\n- Prefer concise correct answers over long vague answers.\n\nOutput JSON ONLY in this exact schema (score must be an integer 0..10):\n\n{\n  \"score\": 0,\n  \"reasons\": [\"...\"],\n  \"flags\": []\n}\n"

# =========================
# IO helpers
# =========================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# =========================
# Ollama generate (no retries)
# =========================
def wrap_prompt(user_prompt: str) -> str:
    return (
        "You are an AI assistant.\n\n"
        "User prompt:\n"
        f"{user_prompt}\n\n"
        "Provide the best possible answer."
    )

def call_generate_once(model: str, prompt: str) -> Tuple[Optional[str], Dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": GEN_TEMPERATURE},
    }
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=GEN_TIMEOUT_SEC)
        latency_ms = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            return None, {"latency_ms": latency_ms, "error": f"HTTP_{r.status_code}: {r.text[:500]}"}
        data = r.json()
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        return None, {"latency_ms": latency_ms, "error": f"REQUEST_EXCEPTION: {str(e)}"}

    resp = (data.get("response") or "").strip()
    meta = {
        "latency_ms": latency_ms,
        "input_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "total_tokens": (
            (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
            if (data.get("prompt_eval_count") is not None or data.get("eval_count") is not None)
            else None
        ),
        "done": data.get("done"),
        "done_reason": data.get("done_reason"),
        "error": None,
    }
    return resp, meta

# =========================
# Judge single (no retries)
# =========================
def extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    blob = text[i:j+1]
    try:
        return json.loads(blob)
    except Exception:
        return None

def clamp_int(v: Any, lo: int, hi: int) -> Optional[int]:
    if not isinstance(v, (int, float)):
        return None
    v = int(round(v))
    return max(lo, min(hi, v))

def judge_single_once(user_prompt: str, answer: str) -> Dict[str, Any]:
    judge_input = (
        JUDGE_SINGLE_PROMPT
        + "\n\nUSER PROMPT:\n" + (user_prompt or "")
        + "\n\nASSISTANT ANSWER:\n" + (answer or "")
        + "\n\nReturn JSON only."
    )

    payload = {
        "model": JUDGE_MODEL,
        "prompt": judge_input,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    t0 = time.time()
    try:
        r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=JUDGE_TIMEOUT_SEC)
        latency_ms = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            return {"raw": None, "parsed": None, "error": f"HTTP_{r.status_code}: {r.text[:500]}", "latency_ms": latency_ms}
        raw = r.json().get("response", "")
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        return {"raw": None, "parsed": None, "error": f"REQUEST_EXCEPTION: {str(e)}", "latency_ms": latency_ms}

    parsed = extract_json_blob(raw)
    return {"raw": raw, "parsed": parsed, "error": None, "latency_ms": latency_ms}

def parse_single_score(judge_parsed: Any) -> Tuple[Optional[int], List[str]]:
    flags: List[str] = []
    if not isinstance(judge_parsed, dict):
        return None, flags
    score = clamp_int(judge_parsed.get("score"), 0, 10)
    fl = judge_parsed.get("flags")
    if isinstance(fl, list):
        flags = [str(x) for x in fl][:10]
    return score, flags

# =========================
# MAIN
# =========================
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tests = load_json(TEST_PROMPTS_FILE)

    out_path = os.path.join(OUT_DIR, "baseline_mistral_only.json")
    summary_path = os.path.join(OUT_DIR, "baseline_mistral_only_summary.json")

    results: List[Dict[str, Any]] = []
    if os.path.exists(out_path):
        try:
            results = load_json(out_path)
        except Exception:
            results = []

    done = {r.get("prompt_id") for r in results if r.get("prompt_id") is not None}

    total_latency: List[float] = []
    total_cost: List[float] = []
    total_quality: List[float] = []

    rng = random.Random(RANDOM_SEED)

    for idx, item in enumerate(tests, start=1):
        pid = item.get("prompt_id")
        prompt = item.get("prompt", "")

        if pid is None or pid in done:
            continue

        selected_model = 'mistral'

        print(f"[{idx}/{len(tests)}] prompt_id={pid} | model={selected_model}", flush=True)

        resp, meta = call_generate_once(selected_model, wrap_prompt(prompt))
        executed = {
            selected_model: {
                "response": resp,
                **meta
            }
        }

        # If generation failed/empty, skip judge and score 0 (consistent & prevents judge stall)
        if (meta.get("error") is not None) or (not (resp or "").strip()):
            judge_info = {
                "raw": None,
                "parsed": {"score": 0, "reasons": ["generation failed/empty"], "flags": ["gen_failed_or_empty"]},
                "error": meta.get("error") or "EMPTY_RESPONSE",
            }
            chosen_score = 0
            chosen_flags = ["gen_failed_or_empty"]
        else:
            judge_info = judge_single_once(prompt, resp)
            chosen_score, chosen_flags = parse_single_score(judge_info.get("parsed"))

            if chosen_score is None:
                # parse failed -> treat as 0 so metrics stay defined
                chosen_score = 0
                chosen_flags = chosen_flags + ["judge_parse_failed"]

        total_latency.append(float(meta.get("latency_ms") or 0))
        total_cost.append(float(meta.get("total_tokens") or 0))
        total_quality.append(float(chosen_score))

        record = {
            "prompt_id": pid,
            "prompt": prompt,
            "baseline": "baseline_mistral_only",
            "selected_model": selected_model,
            "executed": executed,
            "judge": {
                "parsed": judge_info.get("parsed"),
                "raw": judge_info.get("raw"),
                "error": judge_info.get("error"),
                "flags": chosen_flags,
            },
            "selected_quality_score": chosen_score,
        }

        results.append(record)
        done.add(pid)

        if len(results) % CHECKPOINT_EVERY == 0:
            save_json_atomic(out_path, results)

        time.sleep(SLEEP_BETWEEN)

    save_json_atomic(out_path, results)

    summary = {
        "baseline": "baseline_mistral_only",
        "num_prompts": len([r for r in results if r.get("selected_model")]),
        "avg_latency_ms": (sum(total_latency) / len(total_latency)) if total_latency else None,
        "avg_total_tokens_cost": (sum(total_cost) / len(total_cost)) if total_cost else None,
        "avg_quality_score": (sum(total_quality) / len(total_quality)) if total_quality else None,
    }
    save_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
