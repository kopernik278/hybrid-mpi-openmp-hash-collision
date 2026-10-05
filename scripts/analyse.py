#!/usr/bin/env python3
"""analyse.py - figures and LaTeX tables for the report from results/*.jsonl.

CITS3402/CITS5507 Assignment 2 (2026), Shaoming Wu (24914408).  Every platform found in the results
(m4, kaya, setonix, ...) gets its own panel/rows, so re-running the Slurm
jobs and this script refreshes the report automatically.

    python3 scripts/analyse.py            # writes report/figures/*.pdf, report/tables/*.tex
"""
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "report", "figures")
TAB = os.path.join(ROOT, "report", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Categorical slots in fixed order (validated: adjacent CVD dE >= 9), markers as
# secondary encoding for print / colour-vision deficiency.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
MK = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PLAT_ORDER = ["setonix", "kaya", "m4"]
PLAT_NAME = {"setonix": "Setonix", "kaya": "Kaya", "m4": "Apple M4 (dev.)"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.5, "lines.markersize": 4.5,
    "legend.frameon": False, "font.family": "sans-serif", "pdf.fonttype": 42,
})


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["cores"] = df["ranks"] * df["threads"]
    return df


def platforms(df):
    present = list(df["platform"].unique())
    return [p for p in PLAT_ORDER if p in present] + [p for p in present if p not in PLAT_ORDER]


def line(ax, x, y, i, label):
    ax.plot(x, y, color=C[i], marker=MK[i], label=label, markeredgecolor="white",
            markeredgewidth=0.8, solid_capstyle="round")


def plain_log2(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


def fmt_rate(v):
    if v >= 1e6:
        return f"{v / 1e6:.1f}\\,M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}\\,k"
    return f"{v:.0f}"


# ---------------------------------------------------------------- scaling
def scaling(df):
    s = df[df.exp == "scale"]
    if s.empty:
        return
    plats = platforms(s)
    fig, axes = plt.subplots(1, len(plats), figsize=(3.2 * len(plats), 2.2), squeeze=False)
    rows = []
    for ax, plat in zip(axes[0], plats):
        d = s[s.platform == plat]
        base = d[d.impl == "serial"].throughput.mean()
        series = [("omp", "OpenMP (1 process)"), ("mpi", "Pure MPI"), ("hybrid", "Hybrid (best split)")]
        maxc = d.cores.max()
        ax.plot([1, maxc], [1, maxc], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), label="Ideal")
        for i, (impl, lab) in enumerate(series):
            e = d[d.impl == impl]
            if e.empty:
                continue
            g = e.groupby("cores").throughput.max().sort_index()
            line(ax, g.index, g.values / base, i, lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        plain_log2(ax, "x")
        plain_log2(ax, "y")
        ax.set_xlabel("cores")
        ax.set_ylabel("speedup vs serial")
        ax.set_title(PLAT_NAME.get(plat, plat))
        ax.legend(loc="upper left")
        # table rows: best configuration per (impl, cores)
        for impl in ["serial", "omp", "mpi", "hybrid"]:
            e = d[d.impl == impl]
            for c, grp in e.groupby("cores"):
                b = grp.loc[grp.throughput.idxmax()]
                sp = b.throughput / base
                rows.append((PLAT_NAME.get(plat, plat), impl, int(b.nodes), int(b.ranks), int(b.threads),
                             int(c), b.throughput, sp, sp / c))
    save(fig, "scaling.pdf")
    t = pd.DataFrame(rows, columns=["plat", "impl", "nodes", "ranks", "threads", "cores", "thr", "S", "E"])
    # table: serial baseline plus every full-node configuration (best split per impl)
    names = {"serial": "serial", "omp": "OpenMP", "mpi": "MPI", "hybrid": "hybrid"}
    with open(os.path.join(TAB, "scaling.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
                "program & nodes & ranks$\\times$thr. & cores & $X$ & $S$ & $E$\\\\\n")
        for plat, g in t.groupby("plat", sort=False):
            cpn = g[g.nodes == 1].cores.max()
            keep = g[(g.impl == "serial") | (g.cores == g.nodes * cpn)]
            f.write(f"\\midrule\\multicolumn{{7}}{{l}}{{\\textit{{{plat}}}, {cpn} cores per node}}\\\\\n")
            keep = keep.assign(o=keep.impl.map({"serial": 0, "omp": 1, "mpi": 2, "hybrid": 3}))
            for _, r in keep.sort_values(["cores", "o"]).iterrows():
                f.write(f"{names[r.impl]} & {r.nodes} & {r.ranks}$\\times${r.threads} & {r.cores} & "
                        f"{fmt_rate(r.thr)} & {r.S:.1f} & {100 * r.E:.0f}\\%\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    t.to_csv(os.path.join(TAB, "scaling.csv"), index=False)


# ---------------------------------------------------------------- hybrid splits
def hybrid(df):
    s = df[(df.exp == "scale") & (df.impl.isin(["mpi", "hybrid"]))]
    if s.empty:
        return
    plats = platforms(s)
    fig, axes = plt.subplots(1, len(plats), figsize=(3.2 * len(plats), 2.2), squeeze=False)
    for ax, plat in zip(axes[0], plats):
        d = s[s.platform == plat]
        for i, n in enumerate(sorted(d.nodes.unique())):
            full = d[(d.nodes == n) & (d.cores == d[d.nodes == n].cores.max())]
            if full.empty:
                continue
            ref = full[(full.impl == "mpi") & (full.threads == 1)].throughput
            g = full.groupby("threads").throughput.max().sort_index()
            ref = ref.iloc[0] if len(ref) else g.iloc[0]
            line(ax, g.index, g.values / ref, i, f"{n} node{'s' if n > 1 else ''} ({int(full.cores.max())} cores)")
        ax.set_xscale("log", base=2)
        plain_log2(ax, "x")
        ax.axhline(1.0, color=INK2, linewidth=0.8)
        ax.set_ylim(0.5, 1.15)
        ax.set_xlabel("OpenMP threads per MPI process")
        ax.set_ylabel("throughput / pure MPI")
        ax.set_title(PLAT_NAME.get(plat, plat))
        ax.legend(loc="lower left")
    save(fig, "hybrid.pdf")


# ---------------------------------------------------------------- batch size
def batch(df):
    b = df[df.exp == "batch"]
    srch = df[df.exp == "search"]
    if b.empty and srch.empty:
        return
    plats = platforms(pd.concat([b, srch]))
    fig, axes = plt.subplots(len(plats), 3, figsize=(7.2, 2.35 * len(plats)), squeeze=False)
    fig.subplots_adjust(wspace=0.42, hspace=0.75)
    rows = []
    for r, plat in enumerate(plats):
        d = b[b.platform == plat]
        a0, a1, a2 = axes[r]
        sets = [(d[(d.impl == "mpi") & (d.kernel == "full")], "MPI, full kernel"),
                (d[(d.impl == "hybrid") & (d.kernel == "full")], "Hybrid, full kernel"),
                (d[(d.impl == "mpi") & (d.kernel == "fast")], "MPI, fast kernel")]
        for i, (e, lab) in enumerate(sets):
            if e.empty:
                continue
            e = e.sort_values("batch")
            # best of repeated runs per batch size
            e = e.groupby("batch", as_index=False).agg(throughput=("throughput", "max"),
                                                        t_comm_max=("t_comm_max", "min"),
                                                        elapsed=("elapsed", "min"), msgs=("msgs", "first"),
                                                        ranks=("ranks", "first"), threads=("threads", "first"))
            line(a0, e.batch, e.throughput / e.throughput.max(), i, lab)
            line(a1, e.batch, 100 * e.t_comm_max / e.elapsed, i, lab)
            for _, x in e.iterrows():
                rows.append((PLAT_NAME.get(plat, plat), lab, int(x.ranks), int(x.threads), int(x.batch),
                             int(x.msgs), x.throughput, 100 * x.t_comm_max / x.elapsed))
        a0.set_ylabel(f"{PLAT_NAME.get(plat, plat)}\nthroughput / best")
        a1.set_ylabel("% of time in comm.")
        e = srch[srch.platform == plat]
        if not e.empty:
            best = e.t_stop.min()
            for i, fl in enumerate(sorted(e.flush_ms.unique())):
                g = e[e.flush_ms == fl].sort_values("batch")
                lab = "size-only batching" if fl == 0 else f"+ {fl:g} ms flush"
                line(a2, g.batch, 100 * (g.t_stop / best - 1), i, lab)
            a2.set_ylabel(f"extra time vs best {best:.0f} s (%)")
            a2.set_ylim(bottom=0, top=max(10, 1.35 * 100 * (e.t_stop.max() / best - 1)))
            a2.legend(loc="upper center")
        for a in (a0, a1, a2):
            a.set_xscale("log", base=4)
            plain_log2(a, "x")
            a.set_xlabel("batch size $B$ (records)")
        a0.set_ylim(0, 1.08)
        a0.set_title("(a) fixed-work throughput")
        a1.set_title("(b) master time routing/MPI/insert")
        a2.set_title("(c) search to 1st collision")
        if r == len(plats) - 1:
            h, l = a0.get_legend_handles_labels()
            fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.36, 0.0))
    save(fig, "batch.pdf")
    pd.DataFrame(rows, columns=["plat", "set", "ranks", "threads", "B", "msgs", "thr", "comm_pct"]).to_csv(
        os.path.join(TAB, "batch.csv"), index=False)


# ---------------------------------------------------------------- K collisions
def expected_trials(K):
    """E[total trials] for K A/B collisions: 2^25 * Gamma(K+1/2)/Gamma(K)."""
    return 2 ** 25 * math.exp(math.lgamma(K + 0.5) - math.lgamma(K))


def kcoll(df):
    kf = df[df.exp == "kfast"]
    kl = df[df.exp == "kfull"]
    if kf.empty and kl.empty:
        return
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    Ks = sorted(set(kf.K) | set(kl.K))
    xs = [k for k in range(1, max(Ks) + 1)]
    a0.plot(xs, [expected_trials(k) / 2 ** 24 for k in xs], color=INK2, linewidth=0.8,
            linestyle=(0, (4, 3)), label=r"theory $2\,\Gamma(K{+}\frac{1}{2})/\Gamma(K)$")
    for i, plat in enumerate(platforms(kf) if not kf.empty else []):
        d = kf[kf.platform == plat]
        a0.scatter(d.K, d.trials / 2 ** 24, s=14, color=C[i], alpha=0.55, edgecolors="none")
        g = d.groupby("K").trials.mean()
        line(a0, g.index, g.values / 2 ** 24, i, f"{PLAT_NAME.get(plat, plat)} (mean of seeds)")
    a0.set_xscale("log", base=2)
    plain_log2(a0, "x")
    a0.set_xlabel("collisions requested $K$")
    a0.set_ylabel(r"trials until stop / $2^{24}$")
    a0.set_title("trials needed (fast kernel)")
    a0.legend(loc="upper left")
    for i, plat in enumerate(platforms(kl) if not kl.empty else []):
        d = kl[kl.platform == plat].sort_values("K")
        line(a1, d.K, d.elapsed, i, PLAT_NAME.get(plat, plat))
    a1.set_xscale("log", base=2)
    plain_log2(a1, "x")
    a1.set_xlabel("collisions requested $K$")
    a1.set_ylabel("search time (s)")
    a1.set_ylim(bottom=0)
    a1.set_title("search time (full kernel, alpha)")
    a1.legend(loc="upper left")
    save(fig, "kcoll.pdf")
    if not kf.empty:
        g = kf.groupby(["platform", "K"]).agg(trials=("trials", "mean"), t=("elapsed", "mean"),
                                               found=("found", "mean")).reset_index()
        g["theory"] = g.K.map(expected_trials)
        g.to_csv(os.path.join(TAB, "kfast.csv"), index=False)


# ---------------------------------------------------------------- per-pair table
def pairs(df):
    k = df[df.exp == "kernel"]
    sv = df[df.exp == "solve"]
    if k.empty and sv.empty:
        return
    names = sorted(set(k.pair) | set(sv.pair))
    plats = platforms(df)
    lines = ["\\begin{tabular}{lr" + "rrrr" * len(plats) + "}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{PLAT_NAME.get(p, p)}}}" for p in plats) + r"\\",
             "pair & KiB & " + " & ".join(["full/s & fast/s & $t_{1}$ (s) & trials (M)"] * len(plats)) + r"\\",
             "\\midrule"]
    for p in names:
        kb = df.loc[df.pair == p, "file_bytes"].iloc[0]
        cells = [p.replace("_", r"\_"), f"{kb / 1024:.0f}"]
        for plat in plats:
            full = k[(k.platform == plat) & (k.pair == p) & (k.kernel == "full")].throughput
            fast = k[(k.platform == plat) & (k.pair == p) & (k.kernel == "fast")].throughput
            cells.append(fmt_rate(full.mean()) if len(full) else "--")
            cells.append(fmt_rate(fast.mean()) if len(fast) else "--")
            s_ = sv[(sv.platform == plat) & (sv.pair == p)]
            cells.append(f"{s_.t_stop.iloc[-1]:.1f}" if len(s_) else "--")
            cells.append(f"{s_.trials.iloc[-1] / 1e6:.1f}" if len(s_) else "--")
        lines.append(" & ".join(cells) + r"\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "pairs.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df = load()
    print(f"{len(df)} results from platforms {platforms(df)}")
    scaling(df)
    hybrid(df)
    batch(df)
    kcoll(df)
    pairs(df)
    print("figures in report/figures, tables in report/tables")


if __name__ == "__main__":
    main()
