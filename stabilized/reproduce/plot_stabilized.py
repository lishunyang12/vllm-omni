"""Plot published latency records; requires matplotlib==3.10.8."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.report / "latency.json").read_text())["results"]
    lookup = {(r["gpu"], r["model"], r["variant"]): r for r in rows}
    models = [("h3_image", "H3-Image · 4096 × 4096"),
              ("h3_video", "H3 video · 1344 × 768 · 124 frames"),
              ("qwen_image", "Qwen-Image · 1024 × 1024"),
              ("wan22", "Wan2.2 · 1280 × 704 · 121 frames")]
    variants = [("pr_bf16", "BF16", "#425466"),
                ("pr_mxfp8", "MXFP8", "#007F88"),
                ("pr_fp8_block", "Blockwise FP8", "#B5651D")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (model, title) in zip(axes.flat, models):
        maximum = 1.0
        ticks, labels = [], []
        for group, gpu in enumerate(("Rubin", "GB200")):
            for index, (variant, label, color) in enumerate(variants):
                y = 6 - group * 4 - index
                ticks.append(y)
                labels.append(gpu + "  " + label)
                row = lookup.get((gpu, model, variant))
                if row is None:
                    ax.text(0, y, "pending", va="center", color="#64748B")
                    continue
                metadata = json.loads((args.report / row["metadata_path"]).read_text())
                samples = [r["seconds"] for r in metadata["requests"] if r["phase"] == "measured"]
                p05, p50, p95 = np.percentile(samples, [5, 50, 95], method="linear")
                assert abs(p50 - row["p50_seconds"]) < 1e-10
                assert abs(p95 - row["p95_seconds"]) < 1e-10
                maximum = max(maximum, p95)
                ax.barh(y, p50, height=.62, color=color)
                ax.errorbar(p50, y, xerr=[[p50-p05], [p95-p50]], fmt="none",
                            ecolor="#172B4D", capsize=3, linewidth=1.2)
                ax.annotate(f"{p50:.2f} s" + (" †" if not row["stable"] else ""),
                            (p95, y), xytext=(7, 0), textcoords="offset points",
                            va="center", fontsize=10)
        ax.set_xlim(0, maximum * 1.30)
        ax.set_ylim(-.7, 6.8)
        ax.set_yticks(ticks, labels)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=13, pad=10, fontweight="bold")
        ax.set_xlabel("Seconds per request — lower is faster", fontsize=10)
        ax.xaxis.grid(True, color="#E2E8F0", linewidth=.7)
        ax.set_axisbelow(True)
    fig.suptitle("Warmed generation latency · four GPUs per system", x=.035,
                 ha="left", y=.99, fontsize=20, fontweight="bold")
    fig.text(.035, .945, "Bars: P50   |   Whiskers: observed P5–P95 range, not a confidence interval",
             fontsize=11, color="#475569")
    image_counts = ", ".join(f"{gpu} n={lookup.get((gpu, 'h3_image', 'pr_bf16'), {}).get('n', 'pending')}"
                             for gpu in ("Rubin", "GB200"))
    fig.text(.035, .035,
             f"Measured requests per mode — H3-Image: {image_counts}; other models: n=20. One prompt/seed per model.\n"
             "† Fails a declared stability check. All timings retained. BF16 activations; DiT quantization only.\n"
             "Offline Omni.generate wall time; excludes initialization and file writing. Quality differences are reported separately.",
             fontsize=10, color="#475569", linespacing=1.5)
    fig.subplots_adjust(left=.17, right=.975, top=.88, bottom=.16, hspace=.45, wspace=.55)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output.with_suffix("."+extension), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
