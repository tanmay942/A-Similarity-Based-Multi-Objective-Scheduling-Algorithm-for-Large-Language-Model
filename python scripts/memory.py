import json, os
from typing import Any, Dict, List, Tuple

EMB_FILE = "prompt_embeddings.json"
JUDG_FILE = os.path.join("ollama_results", "judgments.json")
OUT_FILE = "memory_db.json"

MODELS: Tuple[str, ...] = ("llama3", "qwen2.5:3b-instruct", "mistral")
EXPECTED_DIM = 768


def rjson(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wjson_atomic(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    emb_rows = rjson(EMB_FILE)
    judg_rows = rjson(JUDG_FILE)

    emb: Dict[int, List[float]] = {}
    for x in emb_rows:
        pid, vec = x.get("id"), x.get("embedding")
        if pid is None or not isinstance(vec, list):
            continue
        pid = int(pid)
        if len(vec) != EXPECTED_DIM:
            raise RuntimeError(f"dim mismatch id={pid}: {len(vec)} != {EXPECTED_DIM}")
        emb[pid] = vec

    judg: Dict[int, Dict[str, Any]] = {
        int(j["prompt_id"]): j for j in judg_rows if "prompt_id" in j
    }

    # Only consider prompts present in both embeddings and judgments
    candidate_ids = sorted(emb.keys() & judg.keys())

    def pick(stats: Dict[str, Any], model: str, key: str):
        return (stats.get(model) or {}).get(key)

    def scores_valid(scores: Dict[str, Any]) -> bool:
        """
        Valid if all models have numeric scores in [0,10].
        Filters out None / null / missing / non-numeric.
        """
        for m in MODELS:
            v = scores.get(m, None)
            if not isinstance(v, (int, float)):
                return False
            if v < 0 or v > 10:
                return False
        return True

    out = []
    kept = 0
    dropped_parse_failed = 0
    dropped_bad_scores = 0

    for pid in candidate_ids:
        j = judg[pid]

        # Drop prompts where judge failed (or was marked failed)
        if j.get("judge_parse_failed", False):
            dropped_parse_failed += 1
            continue

        scores = j.get("scores") or {}
        if not scores_valid(scores):
            dropped_bad_scores += 1
            continue

        stats = j.get("stats") or {}

        out.append({
            "id": pid,
            "embedding": emb[pid],
            "scores": scores,
            "latency_ms": {m: pick(stats, m, "latency_ms") for m in MODELS},
            "total_tokens": {m: pick(stats, m, "total_tokens") for m in MODELS},
            "winner": j.get("winner"),
            "flags": j.get("flags") or {},
        })
        kept += 1

    wjson_atomic(OUT_FILE, out)

    miss_j = sorted(emb.keys() - judg.keys())
    miss_e = sorted(judg.keys() - emb.keys())

    print(f"✅ wrote {len(out)} rows to {OUT_FILE}")
    print(f"kept={kept} dropped_parse_failed={dropped_parse_failed} dropped_bad_scores={dropped_bad_scores}")
    print(f"missing_in_judgments={len(miss_j)} missing_in_embeddings={len(miss_e)}")


if __name__ == "__main__":
    main()