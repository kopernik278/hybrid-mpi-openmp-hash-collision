#!/usr/bin/env python3
"""analyse.py - figures and LaTeX tables for the report from results/*.jsonl.

CITS3402/CITS5507 Assignment 2 (2026), Shaoming Wu (24914408).  Every platform
found in the results (setonix, kaya, m4, ...) gets its own panel/rows, so
re-running the Slurm jobs and this script refreshes the report automatically.

    python3 scripts/analyse.py            # writes report/figures/*.pdf, report/tables/*
"""
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "report", "figures")
TAB = os.path.join(ROOT, "report", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Categorical slots in fixed order (validated: adjacent CVD dE >= 9); markers
# are a second encoding for print and colour-vision deficiency.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
MK = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PLAT_ORDER = ["setonix", "kaya", "m4"]
PLAT_NAME = {"setonix": "Setonix", "kaya": "Kaya", "m4": "Apple M4"}

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
    df = df[df.exp != "smoke"].copy()
    df["cores"] = df["ranks"] * df["threads"]
    return df


def platforms(df):
    present = list(df["platform"].unique())
    return [p for p in PLAT_ORDER if p in present] + [p for p in present if p not in PLAT_ORDER]


def name(p):
    return PLAT_NAME.get(p, p)


def line(ax, x, y, i, label, **kw):
    ax.plot(x, y, color=C[i], marker=MK[i], label=label, markeredgecolor="white",
            markeredgewidth=0.8, solid_capstyle="round", **kw)


def plain(ax, axis="x"):
    f = FuncFormatter(lambda v, _: f"{v:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def save(fig, fname):
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)


def fmt_rate(v):
    if v >= 1e6:
        return f"{v / 1e6:.2f}\\,M" if v < 1e7 else f"{v / 1e6:.1f}\\,M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}\\,k"
    return f"{v:.0f}"


def cores_per_node(d):
    return int(d[d.nodes == 1].cores.max())


# ---------------------------------------------------------------- scaling
def scaling(df):
    s = df[df.exp == "scale"]
    if s.empty:
        return
    plats = platforms(s)
    fig, axes = plt.subplots(2, len(plats), figsize=(2.45 * len(plats), 4.3), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.38)
    rows = []
    for col, plat in enumerate(plats):
        d = s[s.platform == plat]
        base = d[d.impl == "serial"].throughput.max()
        cpn = cores_per_node(d)
        # top: speedup vs cores
        ax = axes[0][col]
        maxc = d.cores.max()
        ax.plot([1, maxc], [1, maxc], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), label="ideal")
        for i, (impl, lab) in enumerate([("omp", "OpenMP"), ("mpi", "pure MPI"), ("hybrid", "hybrid (best)")]):
            e = d[d.impl == impl]
            if e.empty:
                continue
            g = e.groupby("cores").throughput.max().sort_index()
            line(ax, g.index, g.values / base, i, lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        plain(ax, "x")
        plain(ax, "y")
        ax.set_xlabel("cores")
        if col == 0:
            ax.set_ylabel("speedup over serial")
        ax.set_title(f"{name(plat)} ({cpn} cores/node)")
        if col == 0:
            h, lab = ax.get_legend_handles_labels()
            fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.955))
        # bottom: efficiency vs threads per process, one line per node count
        ax = axes[1][col]
        for i, n in enumerate(sorted(d.nodes.unique())):
            full = d[(d.nodes == n) & (d.cores == n * cpn) & d.impl.isin(["mpi", "hybrid"])]
            if full.empty:
                continue
            g = full.groupby("threads").throughput.max().sort_index()
            line(ax, g.index, 100 * g.values / base / (n * cpn), i, f"{n} node{'s' if n > 1 else ''}")
        ax.set_xscale("log", base=2)
        plain(ax, "x")
        ax.set_ylim(0, 105)
        ax.set_xlabel("OpenMP threads per MPI process")
        if col == 0:
            ax.set_ylabel("parallel efficiency (%)")
        ax.legend(loc="lower right")
        # table rows: serial, OpenMP on one node, then MPI and best hybrid per node count
        def add(e, label):
            if e.empty:
                return
            b = e.loc[e.throughput.idxmax()]
            sp = b.throughput / base
            rows.append((name(plat), cpn, label, int(b.nodes), int(b.ranks), int(b.threads),
                         int(b.cores), b.throughput, sp, sp / b.cores))
        add(d[d.impl == "serial"], "serial")
        add(d[(d.impl == "omp") & (d.cores == cpn)], "OpenMP")
        for n in sorted(d.nodes.unique()):
            add(d[(d.impl == "mpi") & (d.nodes == n) & (d.cores == n * cpn)], "MPI")
            add(d[(d.impl == "hybrid") & (d.nodes == n) & (d.cores == n * cpn)], "hybrid")
    save(fig, "scaling.pdf")
    t = pd.DataFrame(rows, columns=["plat", "cpn", "impl", "nodes", "ranks", "threads", "cores", "thr", "S", "E"])
    t.to_csv(os.path.join(TAB, "scaling.csv"), index=False)
    with open(os.path.join(TAB, "scaling.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
                "program & nodes & ranks$\\times$thr. & cores & $X$ (trials/s) & $S$ & $E$\\\\\n")
        for plat, g in t.groupby("plat", sort=False):
            f.write(f"\\midrule\\multicolumn{{7}}{{l}}{{\\textit{{{plat}}}}}\\\\\n")
            for _, r in g.iterrows():
                f.write(f"{r.impl} & {r.nodes} & {r.ranks}$\\times${r.threads} & {r.cores} & "
                        f"{fmt_rate(r.thr)} & {r.S:.1f} & {100 * r.E:.0f}\\%\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------- batch size
def batch(df):
    b = df[df.exp == "batch"]
    srch = df[df.exp == "search"]
    # only platforms with a real sweep (>= 4 batch sizes of the full-kernel MPI runs)
    plats = [p for p in platforms(b)
             if b[(b.platform == p) & (b.impl == "mpi") & (b.kernel == "full")].batch.nunique() >= 4]
    if not plats:
        return
    fig, axes = plt.subplots(len(plats), 3, figsize=(7.2, 1.8 * len(plats)), squeeze=False)
    fig.subplots_adjust(wspace=0.42, hspace=0.7)
    rows = []
    for r, plat in enumerate(plats):
        d = b[b.platform == plat]
        a0, a1, a2 = axes[r]
        ranks = int(d[d.impl == "mpi"].ranks.max())
        sets = [(d[(d.impl == "mpi") & (d.kernel == "full")], "MPI, full kernel"),
                (d[(d.impl == "hybrid") & (d.kernel == "full")], "hybrid, full kernel"),
                (d[(d.impl == "mpi") & (d.kernel == "fast")], "MPI, fast kernel")]
        for i, (e, lab) in enumerate(sets):
            if e.empty:
                continue
            e = e.groupby("batch", as_index=False).agg(
                throughput=("throughput", "max"), t_comm_max=("t_comm_max", "min"),
                elapsed=("elapsed", "min"), msgs=("msgs", "first"),
                ranks=("ranks", "first"), threads=("threads", "first"))
            line(a0, e.batch, e.throughput / e.throughput.max(), i, lab)
            line(a1, e.batch, 100 * e.t_comm_max / e.elapsed, i, lab)
            for _, x in e.iterrows():
                rows.append((name(plat), lab, int(x.ranks), int(x.threads), int(x.batch),
                             int(x.msgs), x.throughput, 100 * x.t_comm_max / x.elapsed))
        e = srch[srch.platform == plat]
        for i, fl in enumerate(sorted(e.flush_ms.unique())):
            g = e[e.flush_ms == fl].sort_values("batch")
            lab = "size-only batching" if fl == 0 else f"+ {fl:g} ms flush"
            line(a2, g.batch, g.t_stop, i, lab)
        if not e.empty:
            a2.set_ylim(bottom=0)
            a2.legend(loc="best")
        for a in (a0, a1, a2):
            a.set_xscale("log", base=4)
            plain(a, "x")
            a.set_xlabel("batch size $B$ (records)")
        a0.set_ylim(0, 1.08)
        a0.set_ylabel(f"{name(plat)}, {ranks} proc.\nthroughput / best")
        a1.set_ylabel("% of time in comm.")
        a2.set_ylabel("time to 1st collision (s)")
        if r == 0:
            a0.set_title("(a) fixed-work throughput")
            a1.set_title("(b) master time in routing/MPI")
            a2.set_title("(c) search, $K=1$, fixed seed")
        if r == len(plats) - 1:
            h, lab = a0.get_legend_handles_labels()
            fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.36, 0.02))
    save(fig, "batch.pdf")
    pd.DataFrame(rows, columns=["plat", "set", "ranks", "threads", "B", "msgs", "thr", "comm_pct"]).to_csv(
        os.path.join(TAB, "batch.csv"), index=False)


# ---------------------------------------------------------------- K collisions
def gratio(K):
    """Gamma(K+1/2)/Gamma(K): expected trials for K collisions in units of 2^25."""
    return math.exp(math.lgamma(K + 0.5) - math.lgamma(K))


def kcoll(df):
    kf = df[df.exp == "kfast"]
    kl = df[df.exp == "kfull"]
    if kf.empty and kl.empty:
        return
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.4, 2.1))
    fig.subplots_adjust(wspace=0.35)
    xs = [1 + 0.25 * i for i in range(61)]
    # left: trial counts where the stop latency is negligible (fewest processes)
    if not kf.empty:
        a0.plot(xs, [2 * gratio(k) for k in xs], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)),
                label=r"theory $2\,\Gamma(K{+}\frac{1}{2})/\Gamma(K)$")
        small = kf[kf.ranks * kf.threads == kf.groupby("platform").cores.transform("min")]
        pl = min(platforms(small), key=lambda p: small[small.platform == p].cores.min())
        d = small[small.platform == pl]
        a0.scatter(d.K, d.trials / 2 ** 24, s=12, color=C[0], alpha=0.5, edgecolors="none")
        g = d.groupby("K").trials.mean()
        line(a0, g.index, g.values / 2 ** 24, 0, f"{name(pl)}, {int(d.cores.min())} proc. (mean)")
    a0.set_xscale("log", base=2)
    plain(a0, "x")
    a0.set_xlabel("collisions requested $K$")
    a0.set_ylabel(r"trials until stop / $2^{24}$")
    a0.set_title("(a) trials needed")
    a0.legend(loc="upper left")
    # right: full-kernel search time relative to K = 1
    a1.plot(xs, [gratio(k) / gratio(1) for k in xs], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)),
            label="theory")
    for i, plat in enumerate(platforms(kl) if not kl.empty else []):
        d = kl[kl.platform == plat].sort_values("K")
        t1 = d[d.K == 1].elapsed.iloc[0]
        line(a1, d.K, d.elapsed / t1, i, f"{name(plat)}, {int(d.cores.iloc[0])} cores")
    a1.set_xscale("log", base=2)
    plain(a1, "x")
    a1.set_xlabel("collisions requested $K$")
    a1.set_ylabel("search time / time for $K=1$")
    a1.set_ylim(bottom=0)
    a1.set_title("(b) full-kernel search time")
    a1.legend(loc="upper left")
    save(fig, "kcoll.pdf")
    if not kf.empty:
        g = kf.groupby(["platform", "ranks", "K"]).agg(trials=("trials", "mean"), t=("elapsed", "mean"),
                                                         found=("found", "mean")).reset_index()
        g["theory"] = 2 ** 25 * g.K.map(gratio)
        g.to_csv(os.path.join(TAB, "kfast.csv"), index=False)


# ---------------------------------------------------------------- per-pair table
def pairs(df):
    k = df[(df.exp == "kernel") & (df.kernel == "full")]
    sv = df[df.exp == "solve"]
    if k.empty and sv.empty:
        return
    kp = platforms(k) if not k.empty else []
    sp = platforms(sv) if not sv.empty else []
    names = sorted(set(k.pair) | set(sv.pair))
    lines = ["\\begin{tabular}{lr" + "r" * len(kp) + "r" * len(sp) + "}", "\\toprule",
             " & & " + f"\\multicolumn{{{len(kp)}}}{{c}}{{full kernel, trials/s per core}}"
             + (f" & \\multicolumn{{{len(sp)}}}{{c}}{{time to collision $t_1$ (s)}}" if sp else "") + r"\\",
             "pair & KiB & " + " & ".join(name(p) for p in kp)
             + "".join(f" & {name(p)} ({int(sv[sv.platform == p].cores.iloc[0])} c.)" for p in sp) + r"\\",
             "\\midrule"]
    for p in names:
        kb = df.loc[df.pair == p, "file_bytes"].iloc[0]
        cells = [p.replace("_", r"\_"), f"{kb / 1024:.0f}"]
        for plat in kp:
            v = k[(k.platform == plat) & (k.pair == p)].throughput
            cells.append(fmt_rate(v.mean()) if len(v) else "--")
        for plat in sp:
            s_ = sv[(sv.platform == plat) & (sv.pair == p)]
            cells.append(f"{s_.t_stop.iloc[-1]:.1f}" if len(s_) else "--")
        lines.append(" & ".join(cells) + r"\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "pairs.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df = load()
    print(f"{len(df)} results from platforms {platforms(df)}")
    scaling(df)
    batch(df)
    kcoll(df)
    pairs(df)
    for stale in ("hybrid.pdf",):
        path = os.path.join(FIG, stale)
        if os.path.exists(path):
            os.remove(path)
    print("figures in report/figures, tables in report/tables")


if __name__ == "__main__":
    main()
