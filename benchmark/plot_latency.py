"""Plot the latency sweep CSV as a Figure-3-style pair of panels (TTFT | TPOT vs context).

    python benchmark/plot_latency.py results/latency_llama1b.csv -o results/latency_llama1b.png

Points are the median over timed repeats; whiskers span min..max. Lengths that
OOM'd are listed in the panel. Also prints a summary table with speedup vs dense.
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-4 of the reference palette, in fixed order (never cycled);
# markers give a second, non-color encoding.
STYLE = {
    "dense": dict(label="Dense (FlashAttention)", color="#2a78d6", marker="o"),
    "ssa":   dict(label="SSA (L=16)",             color="#eb6834", marker="s"),
    "hssa":  dict(label="H-SSA (L=4, J=4)",       color="#1baf7a", marker="^"),
    "ref":   dict(label="SSA eager reference",    color="#6250d6", marker="D"),
}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def series_key(row):
    # eager-reference runs of a gist model are plotted as their own series
    return "ref" if row["config"] != "dense" and row["attn"].startswith("ref") else row["config"]


def load(path):
    runs = defaultdict(lambda: defaultdict(lambda: {"ttft": [], "tpot": []}))
    ooms = defaultdict(list)
    gpu = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            key = series_key(row)
            L = int(row["context_len"])
            gpu.add(row["gpu"])
            if row["status"] == "oom":
                ooms[key].append(L)
                continue
            runs[key][L]["ttft"].append(float(row["ttft_ms"]))
            runs[key][L]["tpot"].append(float(row["tpot_ms"]))
    return runs, ooms, ", ".join(sorted(gpu))


def summarize(vals):
    return statistics.median(vals), min(vals), max(vals)


def plot(runs, ooms, gpu, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
    for ax, metric, title in [(axes[0], "ttft", "Prefill (TTFT)"), (axes[1], "tpot", "Decode (TPOT)")]:
        for key in [k for k in STYLE if k in runs]:
            st = STYLE[key]
            Ls = sorted(runs[key])
            med, lo, hi = zip(*(summarize(runs[key][L][metric]) for L in Ls))
            xs = [L / 1024 for L in Ls]
            ax.errorbar(xs, med, yerr=[[m - a for m, a in zip(med, lo)], [b - m for m, b in zip(med, hi)]],
                        color=st["color"], marker=st["marker"], markersize=6, linewidth=2,
                        capsize=0, elinewidth=1, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                        label=st["label"])
            # direct label at the line end (<= 4 series)
            ax.annotate(st["label"].split(" (")[0], (xs[-1], med[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
        notes = [f"{STYLE[k]['label'].split(' (')[0]}: OOM at {min(v) // 1024}K" for k, v in ooms.items() if k in STYLE]
        if notes:
            ax.text(0.98, 0.03, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Context length (K tokens)", fontsize=9, color=MUTED)
        ax.set_ylabel("Latency (ms)", fontsize=9, color=MUTED)
        ticks = sorted({L for k in runs for L in runs[k]} | {L for v in ooms.values() for L in v})
        ax.set_xticks([L / 1024 for L in ticks], [f"{L // 1024}" for L in ticks])
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.margins(x=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.text(0.99, 0.01, f"Llama-3.2-1B, bf16, batch 1 · {gpu}", ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {out}")


def table(runs):
    dense = runs.get("dense", {})
    print(f"\n{'series':<8}{'ctx':>8}{'TTFT ms':>11}{'TPOT ms':>10}{'TTFT x':>9}{'TPOT x':>9}   (x = dense / series)")
    for key in [k for k in STYLE if k in runs]:
        for L in sorted(runs[key]):
            ttft = statistics.median(runs[key][L]["ttft"])
            tpot = statistics.median(runs[key][L]["tpot"])
            if L in dense:
                sx = f"{statistics.median(dense[L]['ttft']) / ttft:9.2f}{statistics.median(dense[L]['tpot']) / tpot:9.2f}"
            else:
                sx = f"{'-':>9}{'-':>9}"
            print(f"{key:<8}{L:>8}{ttft:>11.2f}{tpot:>10.2f}{sx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.csv)[0] + ".png"
    runs, ooms, gpu = load(a.csv)
    plot(runs, ooms, gpu, out)
    table(runs)


if __name__ == "__main__":
    main()
