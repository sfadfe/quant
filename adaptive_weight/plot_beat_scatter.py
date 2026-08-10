#!/usr/bin/env python3
"""L-sweep scatters: tok/s, VRAM, short_qa, swap vs context (2k…24k).

Also plots tensor-only inplace microbench JSON via --inplace-json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Patch, Rectangle

DEFAULT_CHECKPOINTS = (
    2048,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
    18432,
    20480,
    22528,
    24576,
)
_STAMP_RE = re.compile(r"([0-9]{8}T[0-9]{6}Z)$")


def _run_id_from_name(name: str) -> str:
    """UTC stamp only; strips optional prefixes like beat_w4_."""
    m = _STAMP_RE.search(name)
    return m.group(1) if m else name


def _fmt_Lk(n: int) -> str:
    """Axis label: 2048 → 2k, 6144 → 6k, else raw int."""
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)

BASE = {"color": "black", "marker": "o", "size": 90}
OURS = {"color": "red", "marker": "o", "size": 90}
W8 = {"color": "#1f77b4", "marker": "o", "size": 90}
SCATTER_SIZE = 90
LEGEND_MARKERSCALE = 0.37
LEGEND_FONTSIZE = 12
DEFAULT_BASELINE_LABEL = "AWQ-INT4"
DEFAULT_OURS_LABEL = "Adaptive"
DEFAULT_W8_LABEL = "AWQ-INT8"
DEFAULT_PLOTS = ("toks", "vram", "short_qa", "swap")


class _SmallBarLegendHandler(HandlerPatch):
    """Shorter, thinner rectangle marker for bar-chart legends."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        w = width * 0.5
        h = height * 0.2
        x = xdescent + (width - w) * 0.5
        y = ydescent + (height - h) * 0.5
        p = Rectangle(
            (x, y),
            w,
            h,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            linewidth=0,
        )
        self.update_prop(p, orig_handle, legend)
        return [p]


def _load_steps(run_dir: Path) -> list[dict]:
    """Load L-sweep steps; union xfer_steps.json with any raw_*.json steps."""
    by_key: dict[tuple[str, int], dict] = {}

    def _ingest(rows: list) -> None:
        for r in rows:
            if not isinstance(r, dict):
                continue
            pol = r.get("policy")
            ctx = r.get("ctx_len")
            if pol is None or ctx is None:
                continue
            by_key[(str(pol), int(ctx))] = r

    js = run_dir / "xfer_steps.json"
    if js.exists():
        _ingest(json.loads(js.read_text(encoding="utf-8")))
    for raw in sorted(run_dir.glob("raw_*.json")):
        payload = json.loads(raw.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
            _ingest(payload["steps"])
        elif isinstance(payload, list):
            _ingest(payload)
    if by_key:
        return sorted(by_key.values(), key=lambda r: (str(r.get("policy")), int(r["ctx_len"])))

    csv_path = run_dir / "xfer_steps.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no xfer_steps.json/csv or raw_*.json under {run_dir}")
    import csv

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    for r in rows:
        r["ok"] = str(r.get("ok", "")).lower() in ("1", "true", "yes")
        for k in (
            "ctx_len",
            "peak_vram_mib",
            "tok_per_s",
            "swap_s",
            "ttft_s",
            "tail_ppl",
        ):
            if r.get(k) not in (None, ""):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
        for k in ("short_qa_hit", "needle_hit", "dual_resident_hit"):
            if k in r:
                r[k] = str(r[k]).lower() in ("1", "true", "yes")
    return rows


def _pts(steps: list[dict], pol: str, key: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for r in steps:
        if r.get("policy") != pol or not r.get("ok"):
            continue
        v = r.get(key)
        if v is None:
            continue
        if key in ("short_qa_hit", "needle_hit"):
            y = 1.0 if v else 0.0
        else:
            try:
                y = float(v)
            except (TypeError, ValueError):
                continue
        out.append((int(r["ctx_len"]), y))
    out.sort()
    return out


def _populate_scatter_ax(
    ax: plt.Axes,
    *,
    steps: list[dict],
    series: dict[str, dict],
    key: str,
    ylabel: str,
    title: str,
    transfer_t: int | None,
    yfmt: str = "float1",
    ylim: tuple[float, float] | None = None,
    yticks: list | None = None,
    yticklabels: list[str] | None = None,
    annotate_points: bool = True,
    x_label_fmt: str = "Lk",
    show_legend: bool = True,
    legend_order: list[str] | None = None,
) -> set[int]:
    xticks: set[int] = set()
    for pol, st in series.items():
        xy = _pts(steps, pol, key)
        if not xy:
            continue
        xs, ys = zip(*xy)
        xticks.update(int(x) for x in xs)
        ax.scatter(
            xs,
            ys,
            c=st["color"],
            marker=st["marker"],
            label=st["label"],
            s=st.get("size", SCATTER_SIZE),
            zorder=3,
        )
        if annotate_points:
            for x, y in xy:
                if yfmt == "int":
                    txt = f"{y:.0f}"
                elif yfmt == "float2":
                    txt = f"{y:.2f}"
                elif yfmt == "bool":
                    txt = "hit" if y >= 0.5 else "miss"
                else:
                    txt = f"{y:.1f}"
                ax.annotate(
                    txt,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                    color=st["color"],
                )
    if transfer_t is not None and transfer_t > 0:
        ax.axvline(transfer_t, color="#888", ls=":", lw=1.2, label=f"T={transfer_t}")
    ax.set_xlabel("context length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ticks = sorted(xticks) or list(DEFAULT_CHECKPOINTS)
    ax.set_xticks(ticks)
    if x_label_fmt == "raw":
        ax.set_xticklabels([str(t) for t in ticks])
    else:
        ax.set_xticklabels([_fmt_Lk(t) for t in ticks])
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ax.grid(True, alpha=0.3)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if legend_order:
            by_label = dict(zip(labels, handles))
            handles = [by_label[l] for l in legend_order if l in by_label]
            labels = [l for l in legend_order if l in by_label]
        if handles:
            ax.legend(
                handles,
                labels,
                frameon=False,
                fontsize=LEGEND_FONTSIZE,
                markerscale=LEGEND_MARKERSCALE,
            )
    return xticks


def _scatter_vs_L(
    *,
    steps: list[dict],
    series: dict[str, dict],
    key: str,
    ylabel: str,
    title: str,
    out: Path,
    transfer_t: int | None,
    yfmt: str = "float1",
    ylim: tuple[float, float] | None = None,
    yticks: list | None = None,
    yticklabels: list[str] | None = None,
    annotate_points: bool = True,
    x_label_fmt: str = "Lk",
    legend_order: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    _populate_scatter_ax(
        ax,
        steps=steps,
        series=series,
        key=key,
        ylabel=ylabel,
        title=title,
        transfer_t=transfer_t,
        yfmt=yfmt,
        ylim=ylim,
        yticks=yticks,
        yticklabels=yticklabels,
        annotate_points=annotate_points,
        x_label_fmt=x_label_fmt,
        show_legend=True,
        legend_order=legend_order,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_lsweep(
    run_dir: Path,
    out_dir: Path,
    *,
    baseline: str,
    ours: str,
    transfer_t: int | None = None,
    baseline_label: str | None = None,
    ours_label: str | None = None,
    w8: str | None = "fixed_w8",
    w8_label: str | None = None,
    annotate_points: bool = True,
    plots: tuple[str, ...] | None = None,
) -> None:
    steps = _load_steps(run_dir)
    series: dict[str, dict] = {}
    if w8:
        has_w8 = any(r.get("policy") == w8 and r.get("ok") for r in steps)
        if has_w8:
            series[w8] = {**W8, "label": w8_label or DEFAULT_W8_LABEL}
    series[baseline] = {
        **BASE,
        "label": baseline_label or DEFAULT_BASELINE_LABEL,
    }
    series[ours] = {
        **OURS,
        "label": ours_label or DEFAULT_OURS_LABEL,
    }
    bl = series[baseline]["label"]
    ol = series[ours]["label"]
    wl = series[w8]["label"] if w8 and w8 in series else None
    legend_order = [bl, wl, ol] if wl else [bl, ol]
    legend_order = [x for x in legend_order if x]
    want = set(plots or DEFAULT_PLOTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "steps": steps,
        "series": series,
        "transfer_t": transfer_t,
        "annotate_points": annotate_points,
        "legend_order": legend_order,
    }
    written: list[str] = []
    if "toks" in want:
        _scatter_vs_L(
            **common,
            key="tok_per_s",
            ylabel="tok/s",
            title="Tok/s",
            out=out_dir / "toks_scatter.png",
        )
        written.append("toks_scatter.png")
    if "vram" in want:
        _scatter_vs_L(
            **common,
            key="peak_vram_mib",
            ylabel="MiB",
            title="VRAM usage",
            out=out_dir / "vram_scatter.png",
            yfmt="int",
        )
        written.append("vram_scatter.png")
    if "short_qa" in want:
        _scatter_vs_L(
            **common,
            key="short_qa_hit",
            ylabel="short_qa",
            title="short_qa",
            out=out_dir / "short_qa_scatter.png",
            yfmt="bool",
            ylim=(-0.1, 1.15),
            yticks=[0, 1],
            yticklabels=["miss", "hit"],
        )
        written.append("short_qa_scatter.png")
    if "swap" in want:
        _scatter_vs_L(
            **common,
            key="swap_s",
            ylabel="s",
            title="swap",
            out=out_dir / "swap_scatter.png",
            yfmt="float2",
        )
        written.append("swap_scatter.png")
    if want == set(DEFAULT_PLOTS):
        meta = {
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "checkpoints": sorted({int(r["ctx_len"]) for r in steps if r.get("ok")}),
            "baseline": baseline,
            "ours": ours,
            "w8": w8 if w8 and w8 in series else None,
            "transfer_t": transfer_t,
            "metrics": ["tok_per_s", "peak_vram_mib", "short_qa_hit", "swap_s"],
            "plots": written,
        }
        (out_dir / "plot_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"done {out_dir}")


def plot_toks_vram_combined(
    run_dir: Path,
    out: Path,
    *,
    baseline: str,
    ours: str,
    transfer_t: int | None = None,
    baseline_label: str | None = None,
    ours_label: str | None = None,
    w8: str | None = "fixed_w8",
    w8_label: str | None = None,
    annotate_points: bool = True,
) -> None:
    """Side-by-side Tok/s and VRAM subplots (shared legend)."""
    steps = _load_steps(run_dir)
    series: dict[str, dict] = {}
    if w8:
        has_w8 = any(r.get("policy") == w8 and r.get("ok") for r in steps)
        if has_w8:
            series[w8] = {**W8, "label": w8_label or DEFAULT_W8_LABEL}
    series[baseline] = {
        **BASE,
        "label": baseline_label or DEFAULT_BASELINE_LABEL,
    }
    series[ours] = {
        **OURS,
        "label": ours_label or DEFAULT_OURS_LABEL,
    }
    bl = series[baseline]["label"]
    ol = series[ours]["label"]
    wl = series[w8]["label"] if w8 and w8 in series else None
    legend_order = [bl, wl, ol] if wl else [bl, ol]
    legend_order = [x for x in legend_order if x]

    fig, (ax_toks, ax_vram) = plt.subplots(1, 2, figsize=(16.0, 4.4))
    common = {
        "steps": steps,
        "series": series,
        "transfer_t": transfer_t,
        "annotate_points": annotate_points,
        "show_legend": False,
        "legend_order": legend_order,
    }
    _populate_scatter_ax(
        ax_toks,
        **common,
        key="tok_per_s",
        ylabel="tok/s",
        title="Tok/s",
    )
    _populate_scatter_ax(
        ax_vram,
        **common,
        key="peak_vram_mib",
        ylabel="MiB",
        title="VRAM usage",
        yfmt="int",
    )
    handles, labels = ax_toks.get_legend_handles_labels()
    if legend_order:
        by_label = dict(zip(labels, handles))
        handles = [by_label[l] for l in legend_order if l in by_label]
        labels = [l for l in legend_order if l in by_label]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles),
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            markerscale=LEGEND_MARKERSCALE,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_w4_fraction_stacked(
    out: Path,
    *,
    fractions: dict[int, float] | None = None,
    run_dir: Path | None = None,
    ours: str = "hf_mixed_adaptive",
) -> None:
    """Bar chart: W4 layer fraction (%) vs context L."""
    if fractions is None:
        if run_dir is None:
            fractions = {
                2048: 5.6,
                4096: 22.0,
                8192: 53.0,
                12288: 83.0,
                16384: 98.9,
            }
        else:
            steps = _load_steps(run_dir)
            total_linears = 0
            for r in steps:
                if r.get("policy") == ours and r.get("ok"):
                    total_linears = max(
                        total_linears,
                        int(r.get("n_w8_linears") or 0) + int(r.get("n_w4_linears") or 0),
                    )
            if total_linears <= 0:
                total_linears = 252
            fractions = {}
            for r in steps:
                if r.get("policy") != ours or not r.get("ok"):
                    continue
                ctx = int(r["ctx_len"])
                n_w4 = int(r.get("n_w4_linears") or 0)
                fractions[ctx] = round(100.0 * n_w4 / total_linears, 1)

    checkpoints = sorted(fractions.keys())
    labels = [_fmt_Lk(c) for c in checkpoints]
    w4_pct = [fractions[c] for c in checkpoints]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = list(range(len(checkpoints)))
    width = 0.55
    ax.bar(
        x,
        w4_pct,
        width=width,
        color=BASE["color"],
        label="Int4 layer",
        zorder=3,
    )
    for i, p in enumerate(w4_pct):
        if p <= 0:
            continue
        ax.text(
            i,
            p + 2,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=BASE["color"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("context length")
    ax.set_ylabel("%")
    ax.set_ylim(0, 108)
    ax.set_title("W4 fraction")
    ax.grid(True, axis="y", alpha=0.3)
    legend_patch = Patch(facecolor=BASE["color"], edgecolor=BASE["color"], linewidth=0)
    ax.legend(
        handles=[legend_patch],
        labels=["Int4 layer"],
        handler_map={Patch: _SmallBarLegendHandler()},
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_vram_compare(
    run_dir: Path,
    out: Path,
    *,
    baseline: str = "fixed_w4",
    ours: str = "hf_mixed_adaptive",
    transfer_t: int = 4096,
    baseline_label: str = DEFAULT_BASELINE_LABEL,
    ours_label: str = DEFAULT_OURS_LABEL,
    annotate_points: bool = False,
) -> None:
    """Two-series VRAM compare."""
    steps = _load_steps(run_dir)
    series = {
        baseline: {"color": "black", "marker": "o", "label": baseline_label, "size": SCATTER_SIZE},
        ours: {"color": "red", "marker": "o", "label": ours_label, "size": SCATTER_SIZE},
    }
    _scatter_vs_L(
        steps=steps,
        series=series,
        key="peak_vram_mib",
        ylabel="peak VRAM (MiB)",
        title="VRAM usage",
        out=out,
        transfer_t=transfer_t,
        yfmt="int",
        annotate_points=annotate_points,
        x_label_fmt="raw",
        legend_order=[baseline_label, ours_label],
    )


def _scatter_categories(
    *,
    title: str,
    ylabel: str,
    out: Path,
    categories: list[str],
    series: list[tuple[dict, list[float | None]]],
    yfmt: str = "float1",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = list(range(len(categories)))
    for style, ys in series:
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        px, py = zip(*pts)
        ax.scatter(
            px,
            py,
            c=style["color"],
            marker=style["marker"],
            label=style["label"],
            zorder=3,
        )
        for x, y in pts:
            txt = f"{y:.0f}" if yfmt == "int" else f"{y:.2f}"
            ax.annotate(
                txt,
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color=style["color"],
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(categories)
    ax.set_xlabel("method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_inplace_microbench(bench: dict, out_dir: Path, run_id: str) -> None:
    """Tensor-only windowed W8→W4 stall/peak (not the L-sweep VAS bench)."""
    x = bench["xfer"]
    cold = bench["cold_disk_w4_load"]
    dual = float(bench["dual_resident_estimate_gib"])
    stall = float(x["stall_s_mean"])
    cold_s = float(cold["s"])
    engine_reload_s = 12.0
    base_st = {**BASE, "label": "baseline cold / dual"}
    ours_st = {**OURS, "label": "ours windowed W8→W4 (Orin-first)"}

    _scatter_categories(
        title="transfer stall — windowed layerwise vs cold",
        ylabel="stall (s)",
        out=out_dir / "swap_scatter.png",
        categories=[
            "cold W4\ntensor",
            "windowed\nlayerwise K=2",
            "vLLM engine\nreload≈",
            "dual pointer\n(GPU)",
        ],
        series=[
            (base_st, [cold_s, None, engine_reload_s, 0.0]),
            (ours_st, [None, stall, None, None]),
        ],
        yfmt="float2",
    )
    peak_gpu = float(x["peak_gpu_gib_mean"])
    peak_uni = float(x["peak_unified_est_gib_mean"])
    stage = float(x["max_host_stage_gib_mean"])
    _scatter_categories(
        title="peak memory — windowed layerwise vs dual",
        ylabel="GiB",
        out=out_dir / "vram_scatter.png",
        categories=[
            "GPU peak\n(window)",
            "unified est.\nGPU+stage",
            "host stage\nmax",
            "dual W8+W4\nest.",
        ],
        series=[
            ({**BASE, "label": "dual-resident estimate"}, [None, None, None, dual]),
            ({**OURS, "label": "ours windowed (K=2)"}, [peak_gpu, peak_uni, stage, None]),
        ],
        yfmt="float2",
    )
    (out_dir / "plot_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "kind": "inplace_microbench",
                "stall_s_mean": stall,
                "peak_gpu_gib": peak_gpu,
                "note": "Not L-sweep; use plot_lsweep after phase0_xfer_bench",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"done {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="phase0_xfer_bench out dir (xfer_steps.json)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--baseline", default="fixed_w4")
    p.add_argument("--ours", default="hf_mixed_adaptive")
    p.add_argument(
        "--transfer-t",
        type=int,
        default=0,
        help="vertical marker at transfer threshold; 0 disables",
    )
    p.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    p.add_argument("--ours-label", default=DEFAULT_OURS_LABEL)
    p.add_argument("--w8", default="fixed_w8", help="optional fixed-W8 policy id; empty to skip")
    p.add_argument("--w8-label", default=DEFAULT_W8_LABEL)
    p.add_argument(
        "--inplace-json",
        type=Path,
        default=None,
        help="Optional tensor-only inplace microbench JSON",
    )
    p.add_argument(
        "--also-plot-xfer",
        action="store_true",
        help="Also run plot_xfer.py line plots into out_dir/xfer/",
    )
    p.add_argument(
        "--no-annotate",
        action="store_true",
        help="Omit numeric labels above scatter points",
    )
    p.add_argument(
        "--vram-compare-out",
        type=Path,
        default=None,
        help="Also write 2-series baseline-vs-ours VRAM plot (ppt vram_scatter1)",
    )
    p.add_argument(
        "--toks-vram-out",
        type=Path,
        default=None,
        help="Also write Tok/s + VRAM side-by-side subplot figure",
    )
    p.add_argument(
        "--w4-stack-out",
        type=Path,
        default=None,
        help="Also write stacked W8/W4 fraction bar chart",
    )
    p.add_argument(
        "--plots",
        default="",
        help="Comma-separated: toks,vram,short_qa,swap; none skips L-sweep plots",
    )
    args = p.parse_args()

    if args.inplace_json is not None:
        payload = json.loads(args.inplace_json.read_text(encoding="utf-8"))
        bench = payload.get("bench") or payload
        run_id = _run_id_from_name(args.inplace_json.stem)
        out = args.out_dir or Path("figs") / run_id
        plot_inplace_microbench(bench, out, run_id)
        return 0

    if args.run_dir is None:
        p.error("need --run-dir (L-sweep) or --inplace-json (microbench)")

    out = args.out_dir or Path("figs") / _run_id_from_name(args.run_dir.name)
    ours_label = args.ours_label or f"ours {args.ours}"
    w8 = (args.w8 or "").strip() or None
    plots_arg = (args.plots or "").strip()
    if plots_arg.lower() != "none":
        if plots_arg:
            plots = tuple(p.strip() for p in plots_arg.split(",") if p.strip())
        else:
            plots = None
        plot_lsweep(
            args.run_dir,
            out,
            baseline=args.baseline,
            ours=args.ours,
            transfer_t=args.transfer_t,
            baseline_label=args.baseline_label,
            ours_label=ours_label,
            w8=w8,
            w8_label=args.w8_label,
            annotate_points=not args.no_annotate,
            plots=plots,
        )

    if args.vram_compare_out is not None:
        transfer_t = args.transfer_t if args.transfer_t > 0 else 4096
        plot_vram_compare(
            args.run_dir,
            args.vram_compare_out,
            baseline=args.baseline,
            ours=args.ours,
            transfer_t=transfer_t,
            baseline_label=args.baseline_label,
            ours_label=ours_label,
            annotate_points=not args.no_annotate,
        )

    if args.toks_vram_out is not None:
        plot_toks_vram_combined(
            args.run_dir,
            args.toks_vram_out,
            baseline=args.baseline,
            ours=args.ours,
            transfer_t=args.transfer_t if args.transfer_t > 0 else None,
            baseline_label=args.baseline_label,
            ours_label=ours_label,
            w8=w8,
            w8_label=args.w8_label,
            annotate_points=not args.no_annotate,
        )

    if args.w4_stack_out is not None:
        plot_w4_fraction_stacked(
            args.w4_stack_out,
            run_dir=None,
        )

    if args.also_plot_xfer:
        import runpy
        import sys

        plot_xfer = Path(__file__).resolve().parent / "plot_xfer.py"
        sys.argv = [
            str(plot_xfer),
            "--csv",
            str(args.run_dir / "xfer_steps.csv"),
            "--out-dir",
            str(out / "xfer"),
            "--verdict",
            str(args.run_dir / "verdict.json"),
        ]
        runpy.run_path(str(plot_xfer), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
