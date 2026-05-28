import json
import os
import math
import matplotlib.pyplot as plt

# ✅ Your paths
BASELINE_FILES = {
    "Llama3 Only": "online_baseline_runs/baseline_llama3_only.json",
    "Mistral Only": "online_baseline_runs/baseline_mistral_only.json",
    "Qwen Only": "online_baseline_runs/baseline_qwen_only.json",
    "Random Model": "online_baseline_runs/baseline_random_model.json",
    "Always Latency": "online_baseline_runs/baseline_always_latency.json",
    "Always Quality": "online_baseline_runs/baseline_always_quality.json",
    "Weighted (0.33)": "online_baseline_runs/baseline_weighted_033.json",
}

SAVE_DIR = "./plots_split"

# ✅ Makes latency/tokens readable when there are spikes
USE_LOG_LATENCY = True
USE_LOG_TOKENS = True

# Layout
N_COLS = 2  # 2 columns looks clean for 7 plots (4 rows, last empty)
DPI = 250


def load_points(json_path: str) -> dict[int, dict]:
    """
    Returns: {prompt_id: {"quality": q, "latency_ms": l, "total_tokens": t}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts = {}
    for item in data:
        pid = item.get("prompt_id") or item.get("id")
        if pid is None:
            continue
        pid = int(pid)

        # Quality
        q = item.get("selected_quality_score")
        if q is None:
            judge = item.get("judge", {})
            parsed = judge.get("parsed") if isinstance(judge, dict) else None
            if isinstance(parsed, dict):
                q = parsed.get("score")

        # Latency & tokens (selected model)
        latency_ms = None
        total_tokens = None
        executed = item.get("executed", {})
        if isinstance(executed, dict) and executed:
            sel = item.get("selected_model")
            payload = None
            if sel and sel in executed and isinstance(executed[sel], dict):
                payload = executed[sel]
            else:
                _, payload = next(iter(executed.items()))
                if not isinstance(payload, dict):
                    payload = None
            if payload:
                latency_ms = payload.get("latency_ms")
                total_tokens = payload.get("total_tokens")

        pts[pid] = {"quality": q, "latency_ms": latency_ms, "total_tokens": total_tokens}

    return pts


def small_multiples(all_pts: dict, metric: str, title: str, ylabel: str,
                    out_path: str, use_log: bool):
    """
    Plots one subplot per baseline for the given metric.
    """
    names = list(all_pts.keys())
    n = len(names)
    ncols = N_COLS
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 3.2 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    # Determine common ids across all baselines (ensures aligned x-axis)
    common_ids = sorted(set.intersection(*(set(d.keys()) for d in all_pts.values())))
    if not common_ids:
        raise RuntimeError("No common prompt_ids found across baselines. Check your JSON files.")

    # Plot each baseline
    for idx, name in enumerate(names):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        ys = [all_pts[name][i].get(metric) for i in common_ids]

        ax.plot(common_ids, ys)
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.3, alpha=0.6)

        if use_log:
            ax.set_yscale("log")

    # Turn off unused subplot(s)
    total_axes = nrows * ncols
    for idx in range(n, total_axes):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    # Common x label only on bottom row
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("prompt_id")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=DPI)
    plt.show()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_pts = {name: load_points(fp) for name, fp in BASELINE_FILES.items()}

    # Figure 1: Quality
    small_multiples(
        all_pts,
        metric="quality",
        title="Quality vs prompt_id (Small Multiples: one subplot per baseline)",
        ylabel="Quality",
        out_path=os.path.join(SAVE_DIR, "quality_small_multiples.png"),
        use_log=False
    )

    # Figure 2: Latency
    small_multiples(
        all_pts,
        metric="latency_ms",
        title="Latency vs prompt_id (Small Multiples: one subplot per baseline)",
        ylabel="Latency (ms)",
        out_path=os.path.join(SAVE_DIR, "latency_small_multiples.png"),
        use_log=USE_LOG_LATENCY
    )

    # Figure 3: Tokens
    small_multiples(
        all_pts,
        metric="total_tokens",
        title="Total tokens vs prompt_id (Small Multiples: one subplot per baseline)",
        ylabel="Total tokens",
        out_path=os.path.join(SAVE_DIR, "tokens_small_multiples.png"),
        use_log=USE_LOG_TOKENS
    )

    print("Saved plots to:", os.path.abspath(SAVE_DIR))
    print(" - quality_small_multiples.png")
    print(" - latency_small_multiples.png")
    print(" - tokens_small_multiples.png")


if __name__ == "__main__":
    main()