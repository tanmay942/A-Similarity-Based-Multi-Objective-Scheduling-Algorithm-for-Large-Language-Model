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

MEMORY_DB_FILE = "memory_db_all3.json"      # copy next to this script when running locally
TEST_PROMPTS_FILE = "test_500.json"         # [{ "prompt_id":..., "prompt":"..."}, ...]

OUT_DIR = "online_baseline_runs"
K = 5
SIM_THRESHOLD = 0.5

# LLM generation
GEN_TEMPERATURE = 0.7
GEN_TIMEOUT_SEC = 900

# Judge (deterministic)
JUDGE_TIMEOUT_SEC = 900

SLEEP_BETWEEN = 0.05
CHECKPOINT_EVERY = 5          # judged records checkpoint
GEN_CHECKPOINT_EVERY = 5      # generation checkpoint (raw runs)
PLAN_CHECKPOINT_EVERY = 50    # planning checkpoint within a round (optional)

MODELS = ["llama3", "qwen2.5:3b-instruct", "mistral"]

# Planning/execution rounds:
# - We plan a "round" over remaining prompts.
# - Prompts whose best neighbor is still pending are deferred to next round.
MAX_ROUNDS = 10_000  # safety; effectively "until done"

# -------------------------
# Judge prompts
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

JUDGE_PAIR_PROMPT = """You are an impartial evaluator. The two answers are anonymized.

Score each answer from 0 to 10 using:
- Instruction following (0-4)
- Correctness / factuality (0-4)
- Clarity & completeness (0-2)

Rules:
- Do NOT guess which model wrote which answer.
- Do NOT reward writing style unless it improves correctness or clarity.
- Penalize hallucinations and confident wrong statements heavily.
- If an answer is unsafe, refuses incorrectly, is off-topic, or empty, score low.
- Prefer concise correct answers over long vague answers.

Output JSON ONLY in this exact schema:

{
  "scores": {"A": 0, "B": 0},
  "winner": "A|B|tie",
  "reasons": {"A": ["..."], "B": ["..."]},
  "flags": {"A": [], "B": []}
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

def upsert_memory_row(memory_db: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
    rid = row.get("id")
    for i, existing in enumerate(memory_db):
        if existing.get("id") == rid:
            memory_db[i] = row
            return
    memory_db.append(row)

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

def judge_pair(user_prompt: str, answers_ab: Dict[str, str]) -> Dict[str, Any]:
    judge_input = (
        JUDGE_PAIR_PROMPT
        + "\n\nUSER PROMPT:\n" + (user_prompt or "")
        + "\n\nANSWERS:\n"
        + f"\n[A]\n{answers_ab.get('A','')}\n"
        + f"\n[B]\n{answers_ab.get('B','')}\n"
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

def parse_pair_scores(judge_parsed: Any) -> Tuple[Dict[str, Optional[int]], Dict[str, List[str]], Optional[str]]:
    scores = {"A": None, "B": None}
    flags = {"A": [], "B": []}
    winner = None
    if not isinstance(judge_parsed, dict):
        return scores, flags, winner

    sc = judge_parsed.get("scores")
    if isinstance(sc, dict):
        scores["A"] = clamp_int(sc.get("A"), 0, 10)
        scores["B"] = clamp_int(sc.get("B"), 0, 10)

    fl = judge_parsed.get("flags")
    if isinstance(fl, dict):
        for L in ["A", "B"]:
            if isinstance(fl.get(L), list):
                flags[L] = [str(x) for x in fl.get(L)][:10]

    w = judge_parsed.get("winner")
    if w in ["A", "B", "tie"]:
        winner = w
    return scores, flags, winner

# =========================
# Neighbor aggregation + globals
# =========================
def agg_neighbors(neighbors: List[Tuple[float, Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    out = {m: {"avg_score": float("nan"), "avg_latency": float("nan"), "avg_tokens": float("nan")} for m in MODELS}

    for m in MODELS:
        sw_sc = sw_la = sw_tk = 0.0
        w_sc = w_la = w_tk = 0.0

        for sim, row in neighbors:
            sc = (row.get("scores") or {}).get(m)
            la = (row.get("latency_ms") or {}).get(m)
            tk = (row.get("total_tokens") or {}).get(m)

            if isinstance(sc, (int, float)):
                sw_sc += sim * float(sc)
                w_sc += sim
            if isinstance(la, (int, float)):
                sw_la += sim * float(la)
                w_la += sim
            if isinstance(tk, (int, float)):
                sw_tk += sim * float(tk)
                w_tk += sim

        if w_sc > 0:
            out[m]["avg_score"] = sw_sc / w_sc
        if w_la > 0:
            out[m]["avg_latency"] = sw_la / w_la
        if w_tk > 0:
            out[m]["avg_tokens"] = sw_tk / w_tk

    return out

def pick_global_fastest(memory_db: List[Dict[str, Any]]) -> str:
    avg = {}
    for m in MODELS:
        vals = [(row.get("latency_ms") or {}).get(m) for row in memory_db]
        vals = [v for v in vals if isinstance(v, (int, float))]
        avg[m] = sum(vals)/len(vals) if vals else float("inf")
    return min(avg, key=avg.get)

def pick_global_best_quality(memory_db: List[Dict[str, Any]]) -> str:
    avg = {}
    for m in MODELS:
        vals = [(row.get("scores") or {}).get(m) for row in memory_db]
        vals = [v for v in vals if isinstance(v, (int, float))]
        avg[m] = sum(vals)/len(vals) if vals else float("-inf")
    return max(avg, key=avg.get)

def next_best(exclude: str, preference: List[str]) -> str:
    for m in preference:
        if m != exclude:
            return m
    for m in MODELS:
        if m != exclude:
            return m
    return exclude

def cold_start_two_models(memory_db: List[Dict[str, Any]]) -> Tuple[str, str]:
    fastest = pick_global_fastest(memory_db)
    bestq = pick_global_best_quality(memory_db)
    if fastest == bestq:
        bestq = next_best(fastest, ["llama3", "mistral", "qwen2.5:3b-instruct"])
    return fastest, bestq

def compute_global_bounds(memory_db: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    q_vals: List[float] = []
    l_vals: List[float] = []
    t_vals: List[float] = []

    for row in memory_db:
        sc = row.get("scores") or {}
        la = row.get("latency_ms") or {}
        tk = row.get("total_tokens") or {}
        for m in MODELS:
            vq = sc.get(m)
            vl = la.get(m)
            vt = tk.get(m)
            if isinstance(vq, (int, float)):
                q_vals.append(float(vq))
            if isinstance(vl, (int, float)):
                l_vals.append(float(vl))
            if isinstance(vt, (int, float)):
                t_vals.append(float(vt))

    qmin, qmax = (min(q_vals), max(q_vals)) if q_vals else (0.0, 10.0)
    lmin, lmax = (min(l_vals), max(l_vals)) if l_vals else (0.0, 1.0)
    tmin, tmax = (min(t_vals), max(t_vals)) if t_vals else (0.0, 1.0)
    return {"q": (qmin, qmax), "l": (lmin, lmax), "t": (tmin, tmax)}

# =========================
# Memory row builders
# =========================
def build_pending_row(pid: int, embedding: List[float]) -> Dict[str, Any]:
    return {
        "id": pid,
        "embedding": embedding,
        "scores": {m: None for m in MODELS},
        "latency_ms": {m: None for m in MODELS},
        "total_tokens": {m: None for m in MODELS},
        "winner": None,
        "flags": {m: [] for m in MODELS},
        "pending": True,
    }

def build_memory_row_single(
    pid: int,
    embedding: List[float],
    model: str,
    meta: Dict[str, Any],
    score: Optional[int],
    flags: List[str],
) -> Dict[str, Any]:
    scores = {m: None for m in MODELS}
    latency_ms = {m: None for m in MODELS}
    total_tokens = {m: None for m in MODELS}
    flags_map = {m: [] for m in MODELS}

    if model in MODELS:
        scores[model] = score
        latency_ms[model] = meta.get("latency_ms")
        total_tokens[model] = meta.get("total_tokens")
        flags_map[model] = flags or []

    return {
        "id": pid,
        "embedding": embedding,
        "scores": scores,
        "latency_ms": latency_ms,
        "total_tokens": total_tokens,
        "winner": model,
        "flags": flags_map,
        "pending": False,
    }

def build_memory_row_pair(
    pid: int,
    embedding: List[float],
    letter_to_model: Dict[str, str],
    meta_by_letter: Dict[str, Dict[str, Any]],
    scores_by_letter: Dict[str, Optional[int]],
    flags_by_letter: Dict[str, List[str]],
    winner_letter: Optional[str],
) -> Dict[str, Any]:
    scores = {m: None for m in MODELS}
    latency_ms = {m: None for m in MODELS}
    total_tokens = {m: None for m in MODELS}
    flags_map = {m: [] for m in MODELS}

    for L in ["A", "B"]:
        m = letter_to_model.get(L)
        if m not in MODELS:
            continue
        meta = meta_by_letter.get(L, {})
        latency_ms[m] = meta.get("latency_ms")
        total_tokens[m] = meta.get("total_tokens")
        scores[m] = scores_by_letter.get(L)
        flags_map[m] = flags_by_letter.get(L, []) or []

    winner_model = None
    if winner_letter in ["A", "B"]:
        wm = letter_to_model.get(winner_letter)
        winner_model = wm if wm in MODELS else None
    elif winner_letter == "tie":
        winner_model = "tie"

    return {
        "id": pid,
        "embedding": embedding,
        "scores": scores,
        "latency_ms": latency_ms,
        "total_tokens": total_tokens,
        "winner": winner_model,
        "flags": flags_map,
        "pending": False,
    }

# =========================
# Model selection (neighbors) with GLOBAL normalization
# =========================
def select_model_from_neighbors(agg: Dict[str, Dict[str, float]], bounds: Dict[str, Tuple[float, float]]) -> str:
    def is_nan(x: float) -> bool:
        return isinstance(x, float) and math.isnan(x)

    (qmin, qmax) = bounds["q"]
    (lmin, lmax) = bounds["l"]
    (tmin, tmax) = bounds["t"]

    def norm_hi(v: float, vmin: float, vmax: float) -> float:
        return 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)

    def norm_lo(v: float, vmin: float, vmax: float) -> float:
        return 0.5 if vmax == vmin else (vmax - v) / (vmax - vmin)

    best_m, best_s = None, -1e18
    for m in MODELS:
        q = agg[m]["avg_score"]
        l = agg[m]["avg_latency"]
        t = agg[m]["avg_tokens"]

        if is_nan(q):
            q = qmin
        if is_nan(l):
            l = lmax
        if is_nan(t):
            t = tmax

        qn = norm_hi(q, qmin, qmax)
        ln = norm_lo(l, lmin, lmax)
        tn = norm_lo(t, tmin, tmax)

        s = 0.33 * qn + 0.33 * ln + 0.33 * tn
        if s > best_s:
            best_m, best_s = m, s

    return best_m or "llama3"

# =========================
# MAIN (round-based planning -> model-by-model execution -> update memory)
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    memory_db: List[Dict[str, Any]] = load_json(MEMORY_DB_FILE)
    tests: List[Dict[str, Any]] = load_json(TEST_PROMPTS_FILE)

    out_path = os.path.join(OUT_DIR, "baseline_weighted_033.json")
    summary_path = os.path.join(OUT_DIR, "baseline_weighted_033_summary.json")

    # NEW: extra safety files
    runs_path = os.path.join(OUT_DIR, "baseline_weighted_033_runs_partial.json")      # raw generations (resume-safe)
    plans_dir = os.path.join(OUT_DIR, "baseline_weighted_033_round_plans")            # per-round plans

    results: List[Dict[str, Any]] = load_json_safe(out_path, [])
    done = {r.get("prompt_id") for r in results if r.get("prompt_id") is not None}

    total_cost: List[float] = []
    total_latency: List[float] = []
    total_quality: List[float] = []

    bounds = compute_global_bounds(memory_db)

    # Embedding cache so we don't re-embed when prompt deferred to next round
    emb_cache: Dict[int, List[float]] = {}

    remaining: List[Dict[str, Any]] = [
        x for x in tests
        if x.get("prompt_id") is not None and x.get("prompt_id") not in done
    ]

    # NEW: resume-safe runs store (JSON-friendly keys)
    runs_store: Dict[str, Dict[str, Dict[str, Any]]] = load_json_safe(runs_path, {})

    round_no = 0
    while remaining and round_no < MAX_ROUNDS:
        round_no += 1
        print(f"\n================ ROUND {round_no} | remaining={len(remaining)} ================", flush=True)

        planned: List[Dict[str, Any]] = []
        exec_queue: Dict[str, List[int]] = {m: [] for m in MODELS}
        next_remaining: List[Dict[str, Any]] = []

        def is_pending_row(row: Dict[str, Any]) -> bool:
            return bool(row.get("pending", False))

        # ---- Planning pass
        planned_any = False
        for idx, item in enumerate(remaining, start=1):
            pid = item.get("prompt_id")
            prompt = item.get("prompt", "")
            if pid is None or pid in done:
                continue

            if pid not in emb_cache:
                emb_cache[pid] = get_embedding(prompt)
            qemb = emb_cache[pid]

            neighbors = knn(memory_db, qemb, K, SIM_THRESHOLD)
            resolved_neighbors = [(s, row) for (s, row) in neighbors if not is_pending_row(row)]
            pending_neighbors = [(s, row) for (s, row) in neighbors if is_pending_row(row)]

            if resolved_neighbors:
                agg = agg_neighbors(resolved_neighbors)
                selected_model = select_model_from_neighbors(agg, bounds)

                planned.append({
                    "prompt_id": pid,
                    "prompt": prompt,
                    "embedding": qemb,
                    "neighbors": [{"id": n[1].get("id"), "sim": n[0], "pending": bool(n[1].get("pending", False))} for n in neighbors],
                    "used_neighbors": True,
                    "kind": "single",
                    "models_to_run": [selected_model],
                })
                exec_queue[selected_model].append(pid)
                planned_any = True

            elif pending_neighbors:
                next_remaining.append(item)
                continue

            else:
                m1, m2 = cold_start_two_models(memory_db)
                upsert_memory_row(memory_db, build_pending_row(pid, qemb))

                planned.append({
                    "prompt_id": pid,
                    "prompt": prompt,
                    "embedding": qemb,
                    "neighbors": [],
                    "used_neighbors": False,
                    "kind": "pair",
                    "models_to_run": [m1, m2],
                    "pair": {"m1": m1, "m2": m2},
                })
                exec_queue[m1].append(pid)
                exec_queue[m2].append(pid)
                planned_any = True

            # optional mid-planning checkpoint (memory DB changes happen on cold-start pending rows)
            if idx % PLAN_CHECKPOINT_EVERY == 0:
                save_json_atomic(MEMORY_DB_FILE, memory_db)

        # Deadlock protection
        if not planned_any:
            force_item = next_remaining.pop(0)
            pid = force_item.get("prompt_id")
            prompt = force_item.get("prompt", "")
            if pid is None:
                remaining = next_remaining
                continue

            if pid not in emb_cache:
                emb_cache[pid] = get_embedding(prompt)
            qemb = emb_cache[pid]

            m1, m2 = cold_start_two_models(memory_db)
            upsert_memory_row(memory_db, build_pending_row(pid, qemb))

            planned = [{
                "prompt_id": pid,
                "prompt": prompt,
                "embedding": qemb,
                "neighbors": [],
                "used_neighbors": False,
                "kind": "pair",
                "models_to_run": [m1, m2],
                "pair": {"m1": m1, "m2": m2},
            }]
            exec_queue = {m: [] for m in MODELS}
            exec_queue[m1].append(pid)
            exec_queue[m2].append(pid)

        # Save per-round plan (useful for debugging/resume visibility)
        os.makedirs(plans_dir, exist_ok=True)
        save_json_atomic(os.path.join(plans_dir, f"round_{round_no:04d}.json"), planned)

        # ---- Execution pass (model-by-model) with resume + checkpoints
        gen_count = 0
        for model in MODELS:
            queue = exec_queue[model]
            if not queue:
                continue
            print(f"\n🚀 Executing model={model} | prompts={len(queue)}", flush=True)

            for pid in queue:
                pid_str = str(pid)

                # Resume-safe: skip if already generated for this model
                if pid_str in runs_store and model in runs_store[pid_str] and isinstance(runs_store[pid_str][model], dict):
                    continue

                plan_entry = next((p for p in planned if p["prompt_id"] == pid), None)
                if plan_entry is None:
                    continue
                prompt = plan_entry["prompt"]

                try:
                    resp, meta = call_generate(model, wrap_prompt(prompt))
                    run_obj = {"response": resp, **meta}
                except Exception as e:
                    run_obj = {"response": "", "latency_ms": None, "total_tokens": None, "error": f"GEN_FAILED: {str(e)}"}

                runs_store.setdefault(pid_str, {})[model] = run_obj
                gen_count += 1

                # Generation checkpoint: raw runs file
                if gen_count % GEN_CHECKPOINT_EVERY == 0:
                    save_json_atomic(runs_path, runs_store)
                    print(f"💾 Gen checkpoint -> {runs_path} (new_gen={gen_count})", flush=True)

                time.sleep(SLEEP_BETWEEN)

        # Always save at end of round execution
        save_json_atomic(runs_path, runs_store)

        # ---- Judging + finalize records (in planned order)
        for plan_entry in planned:
            pid = plan_entry["prompt_id"]
            pid_str = str(pid)
            prompt = plan_entry["prompt"]
            qemb = plan_entry["embedding"]
            neighbors_for_log = plan_entry.get("neighbors", [])
            used_neighbors = bool(plan_entry.get("used_neighbors", False))
            kind = plan_entry["kind"]

            executed: Dict[str, Any] = {}
            selected_model: Optional[str] = None
            judge_info: Optional[Dict[str, Any]] = None
            chosen_score: Optional[int] = None

            if kind == "single":
                m = plan_entry["models_to_run"][0]
                run = (runs_store.get(pid_str) or {}).get(m) or {}
                executed[m] = run
                resp = (run.get("response") or "").strip()

                judge_info = judge_single(prompt, resp)
                chosen_score, chosen_flags = parse_single_score(judge_info.get("parsed") if judge_info else None)
                if chosen_score is None:
                    chosen_score = 0
                    chosen_flags = (chosen_flags or []) + ["judge_parse_failed"]

                selected_model = m

                total_latency.append(float(run.get("latency_ms") or 0))
                total_cost.append(float(run.get("total_tokens") or 0))
                total_quality.append(float(chosen_score))

                new_row = build_memory_row_single(pid, qemb, m, run, chosen_score, chosen_flags)
                upsert_memory_row(memory_db, new_row)

            else:
                m1 = plan_entry["pair"]["m1"]
                m2 = plan_entry["pair"]["m2"]

                run1 = (runs_store.get(pid_str) or {}).get(m1) or {}
                run2 = (runs_store.get(pid_str) or {}).get(m2) or {}
                resp1 = (run1.get("response") or "").strip()
                resp2 = (run2.get("response") or "").strip()

                executed[m1] = run1
                executed[m2] = run2

                letters = ["A", "B"]
                random.shuffle(letters)
                letter_to_model = {letters[0]: m1, letters[1]: m2}
                answers_ab = {letters[0]: resp1, letters[1]: resp2}
                meta_by_letter = {letters[0]: run1, letters[1]: run2}

                judge_info = judge_pair(prompt, answers_ab)
                scores_by_letter, flags_by_letter, winner_letter = parse_pair_scores(
                    judge_info.get("parsed") if judge_info else None
                )

                if winner_letter in ["A", "B"]:
                    selected_model = letter_to_model[winner_letter]
                    chosen_score = scores_by_letter.get(winner_letter)
                else:
                    sA = scores_by_letter.get("A")
                    sB = scores_by_letter.get("B")
                    if (sA is None) and (sB is None):
                        selected_model = m1
                        chosen_score = None
                    elif sB is None or (sA is not None and sA >= sB):
                        selected_model = letter_to_model["A"]
                        chosen_score = sA
                    else:
                        selected_model = letter_to_model["B"]
                        chosen_score = sB

                lat1 = float(run1.get("latency_ms") or 0)
                lat2 = float(run2.get("latency_ms") or 0)
                total_latency.append(max(lat1, lat2))

                tok1 = float(run1.get("total_tokens") or 0)
                tok2 = float(run2.get("total_tokens") or 0)
                total_cost.append(tok1 + tok2)

                if chosen_score is not None:
                    total_quality.append(float(chosen_score))

                new_row = build_memory_row_pair(
                    pid, qemb, letter_to_model, meta_by_letter, scores_by_letter, flags_by_letter, winner_letter
                )
                upsert_memory_row(memory_db, new_row)

            record = {
                "prompt_id": pid,
                "prompt": prompt,
                "baseline": "baseline_weighted_033",
                "threshold": SIM_THRESHOLD,
                "k": K,
                "used_neighbors": used_neighbors,
                "neighbors": neighbors_for_log,
                "selected_model": selected_model,
                "executed": executed,
                "judge": {
                    "parsed": (judge_info.get("parsed") if judge_info else None),
                    "raw": (judge_info.get("raw") if judge_info else None),
                },
                "selected_quality_score": chosen_score,
                "round": round_no,
                "kind": kind,
            }

            results.append(record)
            done.add(pid)

            # Judged checkpoint + memory checkpoint
            if len(results) % CHECKPOINT_EVERY == 0:
                save_json_atomic(out_path, results)
                save_json_atomic(MEMORY_DB_FILE, memory_db)
                save_json_atomic(runs_path, runs_store)

        # end-of-round save (always)
        save_json_atomic(out_path, results)
        save_json_atomic(MEMORY_DB_FILE, memory_db)
        save_json_atomic(runs_path, runs_store)

        remaining = next_remaining

    # Final save
    save_json_atomic(out_path, results)
    save_json_atomic(MEMORY_DB_FILE, memory_db)
    save_json_atomic(runs_path, runs_store)

    summary = {
        "baseline": "baseline_weighted_033",
        "num_prompts": len([r for r in results if r.get("selected_model")]),
        "avg_latency_ms": (sum(total_latency) / len(total_latency)) if total_latency else None,
        "avg_total_tokens_cost": (sum(total_cost) / len(total_cost)) if total_cost else None,
        "avg_quality_score": (sum(total_quality) / len(total_quality)) if total_quality else None,
        "global_bounds": bounds,
        "runs_partial_file": runs_path,
        "plans_dir": plans_dir,
    }
    save_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()