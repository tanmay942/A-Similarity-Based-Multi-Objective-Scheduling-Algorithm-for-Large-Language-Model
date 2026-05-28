import os, json, time, random
import requests
from collections import defaultdict

# =========================
# CONFIG
# =========================
OLLAMA_URL = "http://localhost:11436/api/generate"
IN_DIR = "ollama_results"
OUT_FILE = os.path.join(IN_DIR, "judgments.json")

JUDGE_MODEL = "llama3"     # can be same as a candidate; we anonymize labels
TIMEOUT_SEC = 900
SLEEP_BETWEEN = 0.05
CHECKPOINT_EVERY = 10

FILES = {
    "llama3": os.path.join(IN_DIR, "results_llama3.json"),
    "qwen2.5:3b-instruct": os.path.join(IN_DIR, "results_qwen2.5:3b-instruct.json"),
    "mistral": os.path.join(IN_DIR, "results_mistral.json"),
}

JUDGE_PROMPT = """You are an impartial evaluator. The answers are anonymized.

Score each answer from 0 to 10 using:
- Instruction following (0-4)
- Correctness / factuality (0-4)
- Clarity & completeness (0-2)

Rules:
- Do NOT guess which model wrote which answer.
- Do NOT reward writing style unless it improves correctness or clarity.
- Penalize hallucinations and confident wrong statements heavily.
- If an answer is unsafe, refuses incorrectly, is off-topic, or empty, score low.
- Prefer concise correct answers over long vague ones.

Output JSON ONLY in this exact schema:

{
  "scores": {"A": 0, "B": 0, "C": 0},
  "winner": "A|B|C|tie",
  "reasons": {"A": ["..."], "B": ["..."], "C": ["..."]},
  "flags": {"A": [], "B": [], "C": []}
}
"""

# =========================
# HELPERS
# =========================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def atomic_save(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def call_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json().get("response", "")

def extract_json(text: str):
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    blob = text[i:j+1]
    try:
        return json.loads(blob)
    except:
        return None

def normalize_scores(scores_dict):
    """
    Accepts either:
      1) {"A": 7, "B": 6, "C": 0}
      2) {"A": {"Instruction following":4,"Correctness/Factuality":4,"Clarity/Completeness":2}, ...}
    Returns:
      {"A": int 0..10, "B": int 0..10, "C": int 0..10} or None
    """
    def to_int_0_10(x: float) -> int:
        return max(0, min(10, int(round(x))))

    out = {}

    for k in ["A", "B", "C"]:
        v = scores_dict.get(k, None)

        # Case 1: already numeric total
        if isinstance(v, (int, float)):
            out[k] = to_int_0_10(float(v))
            continue

        # Case 2: rubric breakdown dict
        if isinstance(v, dict):
            # be tolerant to key variants
            inst = v.get("Instruction following", v.get("Instruction Following"))
            corr = v.get("Correctness/Factuality", v.get("Correctness / factuality", v.get("Correctness")))
            clar = v.get("Clarity/Completeness", v.get("Clarity & completeness", v.get("Clarity")))

            if not all(isinstance(x, (int, float)) for x in [inst, corr, clar]):
                return None

            total = float(inst) + float(corr) + float(clar)
            out[k] = to_int_0_10(total)
            continue

        # Unknown format
        return None

    return out

def group_by_prompt_id(records_by_model):
    grouped = defaultdict(dict)
    for model_name, records in records_by_model.items():
        for rec in records:
            pid = rec.get("prompt_id")
            if pid is None:
                continue
            grouped[pid][model_name] = rec
    return grouped

def run_judge_once(judge_input: str):
    """
    Returns: (raw_text, parsed_json_or_None, scores_abc_or_None)
    scores_abc is normalized dict {"A":int,"B":int,"C":int} if valid, else None
    """
    raw = call_ollama(JUDGE_MODEL, judge_input)
    parsed = extract_json(raw)
    if not (parsed and isinstance(parsed, dict)):
        return raw, None, None
    scores_abc = normalize_scores(parsed.get("scores", {}))
    if not scores_abc:
        return raw, parsed, None
    return raw, parsed, scores_abc

# =========================
# MAIN
# =========================
def main():
    # Load model result files
    records_by_model = {m: load_json(p) for m, p in FILES.items()}
    grouped = group_by_prompt_id(records_by_model)

    # Resume
    judgments = []
    done = set()
    if os.path.exists(OUT_FILE):
        try:
            judgments = load_json(OUT_FILE)
            done = {j["prompt_id"] for j in judgments if "prompt_id" in j}
            print(f"Resuming: already judged {len(done)} prompts", flush=True)
        except:
            judgments, done = [], set()

    prompt_ids = sorted(grouped.keys())
    print(f"Total prompts found: {len(prompt_ids)}", flush=True)

    for idx, pid in enumerate(prompt_ids, start=1):
        if pid in done:
            continue

        rec_any = next(iter(grouped[pid].values()))
        user_prompt = rec_any.get("prompt", "")

        # Gather candidate answers + latency/tokens (for scheduling analysis)
        answers = {}
        stats = {}
        missing = []

        for m in ["llama3", "qwen2.5:3b-instruct", "mistral"]:
            rec = grouped[pid].get(m, {})
            ans = (rec.get("response", "") or "").strip()
            answers[m] = ans
            if not ans:
                missing.append(m)

            stats[m] = {
                "latency_ms": rec.get("latency_ms", None),
                "input_tokens": rec.get("input_tokens", rec.get("prompt_eval_count", None)),
                "output_tokens": rec.get("output_tokens", rec.get("eval_count", None)),
                "total_tokens": rec.get("total_tokens", None),
                "done_reason": rec.get("done_reason", None),
                "error": rec.get("error", None),
                "hit_num_predict_cap": rec.get("hit_num_predict_cap", False),
            }

        # Anonymization mapping
        model_names = ["llama3", "qwen2.5:3b-instruct", "mistral"]
        random.shuffle(model_names)
        mapping = {"A": model_names[0], "B": model_names[1], "C": model_names[2]}

        print(
            f"[{idx}/{len(prompt_ids)}] Judging prompt_id={pid} "
            f"(A={mapping['A']}, B={mapping['B']}, C={mapping['C']})",
            flush=True
        )

        judge_input = (
            JUDGE_PROMPT
            + "\n\nUSER PROMPT:\n" + user_prompt
            + "\n\nANSWERS:\n"
            + f"\n[A]\n{answers[mapping['A']]}\n"
            + f"\n[B]\n{answers[mapping['B']]}\n"
            + f"\n[C]\n{answers[mapping['C']]}\n"
            + "\n\nReturn JSON only."
        )

        raw = None
        parsed = None
        scores_abc = None
        parse_failed = False

        try:
            # Attempt 1
            raw, parsed, scores_abc = run_judge_once(judge_input)

            # ONE retry if parsing/scores invalid
            if scores_abc is None:
                judge_input_retry = (
                    JUDGE_PROMPT
                    + "\n\nIMPORTANT: Output JSON ONLY. No extra text. Follow the schema exactly.\n"
                    + "\n\nUSER PROMPT:\n" + user_prompt
                    + "\n\nANSWERS:\n"
                    + f"\n[A]\n{answers[mapping['A']]}\n"
                    + f"\n[B]\n{answers[mapping['B']]}\n"
                    + f"\n[C]\n{answers[mapping['C']]}\n"
                    + "\n\nReturn JSON only."
                )
                raw, parsed, scores_abc = run_judge_once(judge_input_retry)

            if scores_abc is None:
                parse_failed = True

        except Exception as e:
            parse_failed = True
            raw = f"JUDGE_CALL_FAILED: {str(e)}"
            parsed = None
            scores_abc = None

        # Build final model->score mapping
        model_scores = {"llama3": None, "qwen2.5:3b-instruct": None, "mistral": None}
        model_winner = None
        model_reasons = {"llama3": [], "qwen2.5:3b-instruct": [], "mistral": []}
        model_flags = {"llama3": [], "qwen2.5:3b-instruct": [], "mistral": []}

        # Scores mapping (only if valid)
        if scores_abc is not None:
            for abc, sc in scores_abc.items():
                model_scores[mapping[abc]] = sc

        # Winner/reasons/flags mapping if we at least have parsed JSON
        if parsed and isinstance(parsed, dict):
            w = parsed.get("winner", None)
            if w in ["A", "B", "C"]:
                model_winner = mapping[w]
            elif w == "tie":
                model_winner = "tie"

            reasons = parsed.get("reasons", {}) or {}
            flags = parsed.get("flags", {}) or {}
            for abc in ["A", "B", "C"]:
                m = mapping[abc]
                if isinstance(reasons.get(abc, None), list):
                    model_reasons[m] = reasons[abc][:3]
                if isinstance(flags.get(abc, None), list):
                    model_flags[m] = flags[abc]

        record = {
            "prompt_id": pid,
            "judge_model": JUDGE_MODEL,
            "mapping": mapping,  # keep for audit; remove later if you want
            "scores": model_scores,
            "winner": model_winner,
            "reasons": model_reasons,
            "flags": model_flags,
            "stats": stats,      # latency/tokens/errors for scheduling metrics
            "missing_answers": missing,
            "judge_parse_failed": parse_failed,
            # Keep raw only if parse failed (debugging) to avoid huge files
            "raw_judge_output": raw if parse_failed else None,
        }

        judgments.append(record)
        done.add(pid)

        if len(judgments) % CHECKPOINT_EVERY == 0:
            atomic_save(OUT_FILE, judgments)
            print(f"Checkpoint saved: {OUT_FILE} (records={len(judgments)})", flush=True)

        time.sleep(SLEEP_BETWEEN)

    atomic_save(OUT_FILE, judgments)
    print(f"\nSaved all judgments -> {OUT_FILE}", flush=True)

if __name__ == "__main__":
    main()