"""Plot context-sweep benchmark markdown tables.

Single model (3 subplots: prefill, decode, TTFT, each with NPU + CPU lines):
    python plot.py results/context_sweep_qwen7b.md

Multi-model overlay (compare several sweeps on the same axes):
    python plot.py results/context_sweep_qwen7b.md results/context_sweep_qwen1.5b.md \\
        --labels "Qwen 7B" "Qwen 1.5B" \\
        --out results/compare.png

Output: PNG next to the input .md by default, or at --out.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


# Parse "352.93 +/- 6.22" or "N/A" or "352.93"
_VAL_RE = re.compile(r"([\d.]+)(?:\s*\+/-\s*([\d.]+))?")
# Parse header cell like "ort-qnn prefill (t/s)" or "ort-cpu TTFT (s)"
_HDR_RE = re.compile(r"(\S+)\s+(\w+)\s*\(")


def parse_md_table(path):
    """Return {ctx: {backend: {metric: (median, stdev)}}}."""
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines()
             if l.strip().startswith("|")]
    if len(lines) < 3:
        raise ValueError(f"{path}: table has fewer than 3 rows")

    header = [c.strip() for c in lines[0].strip("|").split("|")]
    data_rows = lines[2:]  # skip header and separator

    out = {}
    for row in data_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        try:
            ctx = int(cells[0])
        except ValueError:
            continue
        out[ctx] = {}
        for i, col in enumerate(header[1:], start=1):
            m = _HDR_RE.match(col)
            if not m:
                continue
            backend, metric = m.group(1), m.group(2).lower()
            vm = _VAL_RE.match(cells[i])
            if vm:
                median = float(vm.group(1))
                stdev = float(vm.group(2)) if vm.group(2) else 0.0
                out[ctx].setdefault(backend, {})[metric] = (median, stdev)
    return out


def plot_sweeps(parsed_files, labels, out_path):
    """2x2 layout: prefill, decode, TTFT (absolute), NPU speedup ratios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    abs_panels = [
        (axes[0][0], "prefill", "Prefill throughput (tokens/s)"),
        (axes[0][1], "decode",  "Decode throughput (tokens/s)"),
        (axes[1][0], "ttft",    "TTFT (seconds)"),
    ]

    backend_style = {
        "ort-qnn": {"color": "tab:orange", "marker": "o", "pretty": "NPU"},
        "ort-cpu": {"color": "tab:blue",   "marker": "s", "pretty": "CPU"},
    }
    linestyles = ["-", "--", ":", "-."]

    # --- absolute-value panels ---
    for ax, metric, title in abs_panels:
        for fi, (parsed, label) in enumerate(zip(parsed_files, labels)):
            ctxs = sorted(parsed.keys())
            for backend, style in backend_style.items():
                if not all(backend in parsed[c] and metric in parsed[c][backend] for c in ctxs):
                    continue
                med = [parsed[c][backend][metric][0] for c in ctxs]
                std = [parsed[c][backend][metric][1] for c in ctxs]
                full_label = f"{style['pretty']}" + (f" {label}" if label else "")
                ax.errorbar(
                    ctxs, med, yerr=std,
                    marker=style["marker"], color=style["color"],
                    linestyle=linestyles[fi % len(linestyles)],
                    label=full_label, capsize=3, markersize=5,
                )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel(title)
        ax.set_title(title.split("(")[0].strip())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # --- speedup panel ---
    # NPU speedup vs CPU. For throughput (prefill/decode): NPU/CPU. For TTFT: CPU/NPU.
    ax = axes[1][1]
    # Note: TTFT speedup == Prefill speedup mathematically (TTFT = ctx/prefill_tps).
    # Plotting both would produce overlapping lines; we only show prefill + decode.
    speedup_metrics = [
        ("prefill", "Prefill (= TTFT)", "tab:red",   "o", lambda n, c: n / c),
        ("decode",  "Decode",           "tab:green", "s", lambda n, c: n / c),
    ]
    for fi, (parsed, label) in enumerate(zip(parsed_files, labels)):
        ctxs = sorted(parsed.keys())
        for metric, pretty, color, marker, ratio_fn in speedup_metrics:
            if not all("ort-qnn" in parsed[c] and "ort-cpu" in parsed[c] and
                       metric in parsed[c]["ort-qnn"] and metric in parsed[c]["ort-cpu"]
                       for c in ctxs):
                continue
            ratios = [
                ratio_fn(parsed[c]["ort-qnn"][metric][0], parsed[c]["ort-cpu"][metric][0])
                for c in ctxs
            ]
            full_label = f"{pretty}" + (f" {label}" if label else "")
            ax.plot(
                ctxs, ratios, marker=marker, color=color,
                linestyle=linestyles[fi % len(linestyles)],
                label=full_label, markersize=5,
            )
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(ax.get_xlim()[1] * 0.5, 1.05, "parity", fontsize=7, color="gray", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("NPU speedup factor (>1 = NPU faster)")
    ax.set_title("NPU speedup vs CPU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Snapdragon X Elite: Qwen 2.5 7B Instruct (int4) context-length sweep",
                 y=1.00, fontsize=12)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_md", nargs="+",
                    help="One or more context_sweep_*.md files")
    ap.add_argument("--out", default=None,
                    help="Output PNG path (default: matches single input, or results/comparison.png)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Per-file labels for the legend (overlay mode)")
    args = ap.parse_args()

    parsed = [parse_md_table(p) for p in args.input_md]

    if args.labels:
        labels = args.labels
    elif len(args.input_md) == 1:
        labels = [""]
    else:
        labels = [Path(p).stem.replace("context_sweep_", "") for p in args.input_md]
    if len(labels) != len(args.input_md):
        ap.error(f"--labels expects {len(args.input_md)} entries, got {len(labels)}")

    if args.out:
        out = args.out
    elif len(args.input_md) == 1:
        out = str(Path(args.input_md[0]).with_suffix(".png"))
    else:
        out = "results/comparison.png"

    plot_sweeps(parsed, labels, out)


if __name__ == "__main__":
    main()
