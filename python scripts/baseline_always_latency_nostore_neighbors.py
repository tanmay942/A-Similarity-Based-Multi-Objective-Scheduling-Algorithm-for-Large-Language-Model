import os, json, time, random, math
import requests
from typing import Dict, Any, List, Optional, Tuple

# =========================
# CONFIG
# =========================
OLLAMA_GEN_URL = "http://localhost:11436/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11436/api/embeddings"

EMBED_MODEL = "nomic-embed-text"
JUDGE_MODEL = "llama3"

MEMORY_DB_FILE = "memory_db_latency.json"
TEST_PROMPTS_FILE = "test_500.json"

OUT_DIR = "online_baseline_runs"
K = 5
SIM_THRESHOLD = 0.5

GEN_TEMPERATURE = 0.7
GEN_TIMEOUT_SEC = 900
JUDGE_TIMEOUT_SEC = 900

SLEEP_BETWEEN = 0.05

# Checkpoints
CHECKPOINT_EVERY = 5              # results checkpoint (judged records)
GEN_CHECKPOINT_EVERY = 5          # generation checkpoint (raw runs)

MODELS = ["llama3", "qwen2.5:3b-instruct", "mistral"]

# -------------------------
# Judge prompt
# -------------------------
JUDGE_SINGLE_PROMPT = """You are an impartial evaluator.

Given a USER PROMPT and a single ASSISTANT ANSWER, score the answer from 0 to 10 using:
- Instruction following (0-4)
- Correctness / factuality (0-4)
- Clarity & completeness (0-2)

Rules:
- Penalize hallucinations and confident wrong statements heavily.
- If the answer is unsafe, refuses incorrectly, is off-topic, or empty, score low.
- Prefer concise correct answers over long vague answers.

Output JSON ONLY in this exact schema (score must be an integer 0..10):

{
  "score": 0,
  "reasons": ["..."],
  "flags": []
}
"""

# =========================
# IO helpers
# =========================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_json_safe(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        return load_json(path)
    except Exception:
        return default

def save_json_atomic(path: str, obj: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# =========================
# Embeddings + similarity
# =========================
def get_embedding(text: str) -> List[float]:
    r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP_{r.status_code}: {r.text}")
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"Bad embedding response: {data}")
    return emb

def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def knn(memory_db: List[Dict[str, Any]], query_emb: List[float], k: int, threshold: float):
    scored = []
    for row in memory_db:
        emb = row.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        s = cosine_sim(query_emb, emb)
        if s >= threshold:
            scored.append((s, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]  # [(sim, row), ...]

# =========================
# Ollama generate + judge
# =========================
def wrap_prompt(user_prompt: str) -> str:
    return (
        "You are an AI assistant.\n\n"
        "User prompt:\n"
        f"{user_prompt}\n\n"
        "Provide the best possible answer."
    )

def call_generate(model: str, prompt: str) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": GEN_TEMPERATURE}
    }
    t0 = time.time()
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=GEN_TIMEOUT_SEC)
    latency_ms = int((time.time() - t0) * 1000)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP_{r.status_code}: {r.text[:500]}")

    data = r.json()
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
        "error": None
    }
    return resp, meta

def extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    blob = text[i:j+1]
    try:
        return json.loads(blob)
    except Exception:
        return None

def judge_single(user_prompt: str, answer: str) -> Dict[str, Any]:
    judge_input = (
        JUDGE_SINGLE_PROMPT
        + "\n\nUSER PROMPT:\n" + (user_prompt or "")
        + "\n\nASSISTANT ANSWER:\n" + (answer or "")
        + "\n\nReturn JSON only."
    )
    r = requests.post(
        OLLAMA_GEN_URL,
        json={"model": JUDGE_MODEL, "prompt": judge_input, "stream": False, "options": {"temperature": 0.0}},
        timeout=JUDGE_TIMEOUT_SEC
    )
    r.raise_for_status()
    raw = r.json().get("response", "")
    parsed = extract_json_blob(raw)
    return {"raw": raw, "parsed": parsed}

def clamp_int(v: Any, lo: int, hi: int) -> Optional[int]:
    if not isinstance(v, (int, float)):
        return None
    v = int(round(v))
    return max(lo, min(hi, v))

def parse_single_score(judge_parsed: Any) -> Tuple[Optional[int], List[str]]:
    flags: List[str] = []
    if not isinstance(judge_parsed, dict):
        return None, flags
    sc = clamp_int(judge_parsed.get("score"), 0, 10)
    fl = judge_parsed.get("flags")
    if isinstance(fl, list):
        flags = [str(x) for x in fl][:10]
    return sc, flags

# =========================
# Latency-only selection
# =========================
def pick_global_fastest(memory_db: List[Dict[str, Any]]) -> str:
    avg = {}
    for m in MODELS:
        vals = [(row.get("latency_ms") or {}).get(m) for row in memory_db]
        vals = [v for v in vals if isinstance(v, (int, float))]
        avg[m] = sum(vals)/len(vals) if vals else float("inf")
    return min(avg, key=avg.get)

def select_model_from_neighbors_latency(neighbors: List[Tuple[float, Dict[str, Any]]]) -> str:
    # similarity-weighted latency: sum(sim*lat)/sum(sim)
    best = None
    best_wlat = float("inf")

    for m in MODELS:
        wsum = 0.0
        lsum = 0.0
        for sim, row in neighbors:
            la = (row.get("latency_ms") or {}).get(m)
            if isinstance(la, (int, float)):
                wsum += float(sim)
                lsum += float(sim) * float(la)
        if wsum <= 0.0:
            continue
        wlat = lsum / wsum
        if best is None or wlat < best_wlat:
            best = m
            best_wlat = wlat

    return best or "qwen2.5:3b-instruct"

# =========================
# MAIN: Decide first -> execute model-by-model (with crash-safe checkpoints)
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    memory_db: List[Dict[str, Any]] = load_json(MEMORY_DB_FILE)
    tests: List[Dict[str, Any]] = load_json(TEST_PROMPTS_FILE)

    out_path = os.path.join(OUT_DIR, "baseline_always_latency.json")
    summary_path = os.path.join(OUT_DIR, "baseline_always_latency_summary.json")

    # NEW: extra safety files
    runs_path = os.path.join(OUT_DIR, "baseline_always_latency_runs_partial.json")
    planned_path = os.path.join(OUT_DIR, "baseline_always_latency_planned.json")

    # Load existing judged results (resume)
    results: List[Dict[str, Any]] = load_json_safe(out_path, [])
    done = {r.get("prompt_id") for r in results if r.get("prompt_id") is not None}

    total_latency: List[float] = []
    total_cost: List[float] = []
    total_quality: List[float] = []

    global_fastest = pick_global_fastest(memory_db)

    # -------- PLANNING (decide model for each prompt)
    planned: List[Dict[str, Any]] = []
    exec_queue: Dict[str, List[int]] = {m: [] for m in MODELS}
    emb_cache: Dict[int, List[float]] = {}

    for idx, item in enumerate(tests, start=1):
        pid = item.get("prompt_id")
        prompt = item.get("prompt", "")
        if pid is None or pid in done:
            continue

        if pid not in emb_cache:
            emb_cache[pid] = get_embedding(prompt)
        qemb = emb_cache[pid]

        neighbors = knn(memory_db, qemb, K, SIM_THRESHOLD)
        used_neighbors = len(neighbors) >= 1

        if used_neighbors:
            selected_model = select_model_from_neighbors_latency(neighbors)
        else:
            selected_model = global_fastest

        planned.append({
            "prompt_id": pid,
            "prompt": prompt,
            "neighbors": [{"id": n[1].get("id"), "sim": n[0]} for n in neighbors],
            "used_neighbors": used_neighbors,
            "selected_model": selected_model
        })
        exec_queue[selected_model].append(pid)

    # Save plan (useful for debugging)
    save_json_atomic(planned_path, planned)

    # -------- EXECUTION (model-by-model) with generation checkpoints + resume
    # JSON-friendly format: runs[str(pid)][model] = {response/meta}
    runs: Dict[str, Dict[str, Dict[str, Any]]] = load_json_safe(runs_path, {})

    gen_count = 0
    for model in MODELS:
        queue = exec_queue[model]
        if not queue:
            continue
        print(f"\n🚀 Executing model={model} | prompts={len(queue)}", flush=True)

        for pid in queue:
            pid_str = str(pid)

            # Resume-safe: skip if already generated
            if pid_str in runs and model in runs[pid_str] and isinstance(runs[pid_str][model], dict):
                continue

            plan_entry = next((p for p in planned if p["prompt_id"] == pid), None)
            if not plan_entry:
                continue
            prompt = plan_entry["prompt"]

            try:
                resp, meta = call_generate(model, wrap_prompt(prompt))
                run_obj = {"response": resp, **meta}
            except Exception as e:
                run_obj = {"response": "", "latency_ms": None, "total_tokens": None, "error": f"GEN_FAILED: {str(e)}"}

            runs.setdefault(pid_str, {})[model] = run_obj
            gen_count += 1

            # checkpoint generation results
            if gen_count % GEN_CHECKPOINT_EVERY == 0:
                save_json_atomic(runs_path, runs)
                print(f"💾 Gen checkpoint saved -> {runs_path} (generated={gen_count})", flush=True)

            time.sleep(SLEEP_BETWEEN)

    # final generation checkpoint
    save_json_atomic(runs_path, runs)

    # -------- JUDGE + SAVE RECORDS (original planned order)
    for i, plan_entry in enumerate(planned, start=1):
        pid = plan_entry["prompt_id"]
        prompt = plan_entry["prompt"]
        selected_model = plan_entry["selected_model"]

        run = (runs.get(str(pid)) or {}).get(selected_model) or {}
        resp = (run.get("response") or "").strip()

        executed = {selected_model: run}

        judge_info = judge_single(prompt, resp)
        chosen_score, chosen_flags = parse_single_score(judge_info.get("parsed") if judge_info else None)
        if chosen_score is None:
            chosen_score = 0
            chosen_flags = (chosen_flags or []) + ["judge_parse_failed"]

        total_latency.append(float(run.get("latency_ms") or 0))
        total_cost.append(float(run.get("total_tokens") or 0))
        total_quality.append(float(chosen_score))

        record = {
            "prompt_id": pid,
            "prompt": prompt,
            "baseline": "baseline_always_latency",
            "threshold": SIM_THRESHOLD,
            "k": K,
            "used_neighbors": plan_entry["used_neighbors"],
            "neighbors": plan_entry["neighbors"],
            "selected_model": selected_model,
            "executed": executed,
            "judge": {
                "parsed": (judge_info.get("parsed") if judge_info else None),
                "raw": (judge_info.get("raw") if judge_info else None),
                "flags": chosen_flags,
            },
            "selected_quality_score": chosen_score,
        }

        results.append(record)
        done.add(pid)

        # checkpoint judged results
        if len(results) % CHECKPOINT_EVERY == 0:
            save_json_atomic(out_path, results)

    save_json_atomic(out_path, results)

    summary = {
        "baseline": "baseline_always_latency",
        "num_prompts": len([r for r in results if r.get("selected_model")]),
        "avg_latency_ms": (sum(total_latency) / len(total_latency)) if total_latency else None,
        "avg_total_tokens_cost": (sum(total_cost) / len(total_cost)) if total_cost else None,
        "avg_quality_score": (sum(total_quality) / len(total_quality)) if total_quality else None,
        "global_fastest": global_fastest,
        "runs_partial_file": runs_path,
        "planned_file": planned_path,
    }
    save_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()