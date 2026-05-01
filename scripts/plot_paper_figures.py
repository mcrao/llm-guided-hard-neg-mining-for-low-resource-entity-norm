#!/usr/bin/env python3
"""
plot_paper_figures.py — Generate Figures 1, 3, and 4 for the NeurIPS paper.
Uses Plotly + kaleido for clean, publication-quality PDF output.

Usage
-----
  uv run python scripts/plot_paper_figures.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR     = PROJECT_ROOT / "manuscript" / "NeurIPS" / "myPaper" / "figures"

# ── Colour palette ────────────────────────────────────────────────────────────
C1   = "#d62728"   # Acc@1 — red
C5   = "#1f77b4"   # Acc@5 — blue
CMRR = "#48A111"   # MRR   — green

BASE_A1  = 0.0433
BASE_A5  = 0.5108
BASE_MRR = 0.2486

FONT = "Arial"


def _save(fig: go.Figure, name: str, w: int, h: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    fig.write_image(str(out), format="pdf", width=w, height=h, scale=2)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def _yax(**kw) -> dict:
    """Common y-axis style: white bg, dotted grid, clean spines."""
    base = dict(
        showgrid=True, gridcolor="#dddddd", gridwidth=1, griddash="dot",
        zeroline=False, showline=True, linecolor="#cccccc", linewidth=1,
        tickfont=dict(family=FONT, size=11),
        title_font=dict(family=FONT, size=12),
    )
    base.update(kw)
    return base


def _xax(**kw) -> dict:
    base = dict(
        showgrid=False, zeroline=False,
        showline=True, linecolor="#cccccc", linewidth=1,
        tickfont=dict(family=FONT, size=11),
        title_font=dict(family=FONT, size=12),
    )
    base.update(kw)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Structural Acc@1 Ceiling
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig1() -> None:
    epoch_df = pd.read_csv(RESULTS_DIR / "epoch_sweep.csv").sort_values("epochs")
    base_df  = (pd.read_csv(RESULTS_DIR / "baselines.csv")
                .query("method == 'bi_encoder_in_batch'")
                .sort_values("n_train_match_pairs"))

    epochs = epoch_df["epochs"].tolist()
    a1     = epoch_df["acc_at_1"].tolist()
    a5     = epoch_df["acc_at_5"].tolist()
    mrr    = epoch_df["mrr"].tolist()
    ref_a1 = a1[0]
    ref_a5 = a5[0]

    b1   = base_df["acc_at_1"].tolist()
    b5   = base_df["acc_at_5"].tolist()
    bmrr = base_df["mrr"].tolist()
    b_x  = ["698 (10%)", "1,744 (25%)", "6,977 (100%)"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
        subplot_titles=[
            "<b>Panel A: Performance vs. Training Epochs</b>",
            "<b>Panel B: Performance vs. Training Data Scale (3 Epochs)</b>",
        ],
        horizontal_spacing=0.14,
    )

    fig.update_annotations(font_size=20, font_family=FONT)

    # Nudge subplot titles upward to open space between title and plot
    for ann in fig.layout.annotations:
        ann.y += 0.03   # ← tune this: 0.03 = subtle, 0.08 = generous

    # ── Panel A — left axis: Acc@1 + MRR ─────────────────────────────────────
    # Epoch 15 label is omitted — the annotation box shows the value there
    a1_labels = [f"<b>{v:.4f}</b>" if i < len(a1) - 1 else ""
                 for i, v in enumerate(a1)]
    fig.add_trace(go.Scatter(
        x=epochs, y=a1, name="Acc@1 (left axis)",
        mode="lines+markers+text",
        line=dict(color=C1, width=2.5),
        marker=dict(symbol="circle", size=8, color=C1),
        text=a1_labels,
        textposition="top center",
        textfont=dict(color=C1, size=16, family=FONT),
        legendgroup="a1",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=epochs, y=mrr, name="MRR (left axis)",
        mode="lines+markers+text",
        line=dict(color=CMRR, width=2.5),
        marker=dict(symbol="triangle-up", size=12, color=CMRR,
                    line=dict(color="white", width=1)),
        text=[f"<b>{v:.4f}</b>" for v in mrr],
        textposition="bottom center",
        textfont=dict(color=CMRR, size=16, family=FONT),
        legendgroup="mrr",
    ), row=1, col=1, secondary_y=False)

    # Panel A — right axis: Acc@5
    fig.add_trace(go.Scatter(
        x=epochs, y=a5, name="Acc@5 (right axis)",
        mode="lines+markers+text",
        line=dict(color=C5, width=2.5),
        marker=dict(symbol="square", size=8, color=C5),
        text=[f"<b>{v:.4f}</b>" for v in a5],
        textposition="top center",
        textfont=dict(color=C5, size=16, family=FONT),
        legendgroup="a5",
    ), row=1, col=1, secondary_y=True)

    # Reference dashes — both as Scatter traces so they appear in the legend
    fig.add_trace(go.Scatter(
        x=[epochs[0], epochs[-1]], y=[ref_a1, ref_a1],
        name=f"Acc@1 3-ep ref. ({ref_a1:.4f})",
        mode="lines", line=dict(dash="dash", color=C1, width=1.5), opacity=0.7,
        legendgroup="ref_a1",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=[epochs[0], epochs[-1]], y=[ref_a5, ref_a5],
        name=f"Acc@5 3-ep ref. ({ref_a5:.4f})",
        mode="lines", line=dict(dash="dash", color=C5, width=1.5), opacity=0.7,
        legendgroup="ref_a5",
    ), row=1, col=1, secondary_y=True)

    # Annotation: Acc@1 drop at epoch 15 (label suppressed above; value shown here)
    fig.add_annotation(
        x=15, y=a1[-1],
        xref="x", yref="y",
        text=f"<b>Acc@1 drops to {a1[-1]:.4f}</b><br>below 3-ep. reference",
        showarrow=True, arrowhead=2, arrowcolor="black", arrowwidth=1.5,
        ax=-110, ay=-100,
        font=dict(color=C1, size=16, family=FONT),
        bgcolor="white", bordercolor=C1, borderwidth=1.5, borderpad=5,
        align="center",
    )

    # Panel A axis config
    fig.update_yaxes(title_text="Performance Metric (Acc@1 | MRR)",
                     range=[0.00, 0.40], tickformat=".2f",
                     secondary_y=False, row=1, col=1,
                     **_yax(tickfont=dict(family=FONT, size=18),
                            title_font=dict(family=FONT, size=18)))
    fig.update_yaxes(title_text="Acc@5", range=[0.30, 0.70], tickformat=".2f",
                     secondary_y=True, row=1, col=1,
                     title_font_color=C5, tickfont=dict(color=C5, family=FONT, size=18),
                     title_font=dict(family=FONT, size=18, color=C5),
                     showgrid=False, zeroline=False,
                     showline=True, linecolor="#cccccc")
    fig.update_xaxes(title_text="Training Epochs", tickvals=epochs,
                     row=1, col=1,
                     **_xax(tickfont=dict(family=FONT, size=18),
                            title_font=dict(family=FONT, size=18)))

    # ── Panel B — all metrics vs data scale ───────────────────────────────────
    for vals, name, color, sym, tpos, group, msize, mborder in [
        (b5,   "Acc@5",  C5,   "square",      "top center",    "a5",  8,  None),
        (bmrr, "MRR",    CMRR, "triangle-up", "bottom center", "mrr", 12, dict(color="white", width=1)),
        (b1,   "Acc@1",  C1,   "circle",      "top center",    "a1",  8,  None),
    ]:
        mkr = dict(symbol=sym, size=msize, color=color)
        if mborder:
            mkr["line"] = mborder
        fig.add_trace(go.Scatter(
            x=b_x, y=vals, name=name,
            mode="lines+markers+text",
            line=dict(color=color, width=2.5),
            marker=mkr,
            text=[f"<b>{v:.4f}</b>" for v in vals],
            textposition=tpos,
            textfont=dict(color=color, size=16, family=FONT),
            legendgroup=group,
            showlegend=False,   # already in legend from Panel A
        ), row=1, col=2)

    fig.update_yaxes(title_text="Performance Metric",
                     range=[0.00, 0.70], tickformat=".2f",
                     row=1, col=2,
                     **_yax(tickfont=dict(family=FONT, size=18),
                            title_font=dict(family=FONT, size=18)))
    fig.update_xaxes(title_text="Training Match Pairs (Data Scale)",
                     row=1, col=2,
                     **_xax(tickfont=dict(family=FONT, size=18),
                            title_font=dict(family=FONT, size=18)))

    # ── Footer note boxes ─────────────────────────────────────────────────────
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=-0.29,
        text=(
            "● <b>Observation:</b> Acc@5 and MRR consistently improve with more training"
            " (epochs or data), indicating better overall ranking quality."
        ),
        showarrow=False,
        font=dict(size=21, color="#333333", family=FONT),
        bgcolor="#f8f8f8", bordercolor="#aaaaaa", borderwidth=1, borderpad=10,
        align="center",
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=-0.42,
        text=(
            "★ <b>Key Finding:</b> Acc@1 remains essentially flat (~4.5%),"
            " a persistent structural ceiling that standard scaling cannot break."
        ),
        showarrow=False,
        font=dict(size=21, color=C1, family=FONT),
        bgcolor="#fff0f0", bordercolor=C1, borderwidth=1, borderpad=10,
        align="center",
    )

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>Figure 1: Empirical Demonstration of the Structural Acc@1 Ceiling</b>",
            font=dict(size=27, family=FONT, color="#111111"),
            x=0.5, xanchor="center",
            pad=dict(b=24),
        ),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.10, yanchor="bottom",
            font=dict(size=18, family=FONT),
            bgcolor="white", bordercolor="#cccccc", borderwidth=1,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=13),
        margin=dict(l=75, r=75, t=185, b=220),
    )

    _save(fig, "fig1_ceiling.pdf", w=1400, h=820)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Five-Factor Ablation Results
# ═══════════════════════════════════════════════════════════════════════════════

def _add_bar_panel(
    fig: go.Figure,
    rows_data: list[tuple],   # [(label, acc1, acc5, mrr), ...]
    row: int,
    col: int,
    show_legend: bool,
    annotations: list[tuple] | None = None,
    # annotations: [(text, x_cat, y_val, ax_px, ay_px, color), ...]
    c1: str = C1,
    c5: str = C5,
    cmrr: str = CMRR,
    label_sz: int = 11,
    annot_sz: int = 11,
    suppress_text_at: dict | None = None,
    # suppress_text_at: {"Acc@1": ["CoT"], "Acc@5": ["4:1"]} — blanks bar labels
    # that would collide with annotation arrowheads; value shown in annotation instead
) -> None:
    x_labels = [r[0] for r in rows_data]

    for vals, name, color, grp in [
        ([r[1] for r in rows_data], "Acc@1", c1,   "grp_a1"),
        ([r[2] for r in rows_data], "Acc@5", c5,   "grp_a5"),
        ([r[3] for r in rows_data], "MRR",   cmrr, "grp_mrr"),
    ]:
        texts = [f"<b>{v:.3f}</b>" for v in vals]
        if suppress_text_at and name in suppress_text_at:
            for lbl in suppress_text_at[name]:
                if lbl in x_labels:
                    texts[x_labels.index(lbl)] = ""
        fig.add_trace(go.Bar(
            x=x_labels, y=vals, name=name,
            marker=dict(color=color, opacity=0.90,
                        line=dict(color="white", width=1.0)),
            text=texts,
            textposition="outside",
            textfont=dict(size=label_sz, color=color, family=FONT),
            legendgroup=grp,
            showlegend=show_legend,
            cliponaxis=False,
        ), row=row, col=col)

    # Baseline dashed reference lines
    for yval, color in [(BASE_A1, c1), (BASE_A5, c5), (BASE_MRR, cmrr)]:
        fig.add_hline(y=yval, line_dash="dot", line_color=color,
                      line_width=1.5, opacity=0.6, row=row, col=col)

    # Derive per-subplot axis references so ax/ay can be given in DATA coordinates
    n = (row - 1) * 2 + col
    xax = "x" if n == 1 else f"x{n}"
    yax = "y" if n == 1 else f"y{n}"

    if annotations:
        # tuple: (text, x_arrow, y_arrow, ax_textbox, ay_textbox, color)
        # ax_textbox / ay_textbox are DATA coordinates (not pixels)
        for text, x_cat, y_val, ax_data, ay_data, color in annotations:
            fig.add_annotation(
                x=x_cat, y=y_val,
                text=text,
                showarrow=True, arrowhead=2,
                arrowcolor=color, arrowwidth=1.5,
                ax=ax_data, ay=ay_data,
                axref=xax, ayref=yax,
                font=dict(size=annot_sz, color=color, family=FONT),
                bgcolor="white", bordercolor=color,
                borderwidth=1.5, borderpad=7,
                align="center",
                row=row, col=col,
            )

    fig.update_yaxes(range=[0, 0.85], tickformat=".2f",
                     title_text="Metric Value",
                     row=row, col=col,
                     **_yax(tickfont=dict(family=FONT, size=18),
                            title_font=dict(family=FONT, size=18)))
    fig.update_xaxes(row=row, col=col,
                     **_xax(tickfont=dict(family=FONT, size=18),
                            title_font=dict(family=FONT, size=18)))


def plot_fig3() -> None:
    # User-specified palette
    BC1   = "#744577"   # plum-purple   → Acc@1
    BC5   = "#2D3C59"   # navy-blue  → Acc@5.
    BCMRR = "#574964"   # sage-green    → MRR

    df = pd.read_csv(RESULTS_DIR / "m3_results.csv")
    base_row = ("Baseline", BASE_A1, BASE_A5, BASE_MRR)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "<b>A. A3 Prompting Strategy</b><br>"
            "<sup>(type=component_swap, backbone=distilbert, ratio=1:1)</sup>",
            "<b>B. A4 Negative Type</b><br>"
            "<sup>(strategy=CoT, backbone=distilbert, ratio=1:1)</sup>",
            "<b>C. A5 LLM-gen:in-batch Ratio</b><br>"
            "<sup>(type=component_swap, strategy=CoT, backbone=distilbert)</sup>",
            "<b>D. A1 Backbone Sensitivity</b><br>"
            "<sup>(type=component_swap, strategy=CoT, ratio=1:1)</sup>",
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
    )
    fig.update_layout(barmode="group")

    # Bump subplot title fonts (layout annotations added by make_subplots)
    fig.update_annotations(font_size=24, font_family=FONT)

    # ── A. A3 Prompting Strategy ──────────────────────────────────────────────
    a3 = (df[(df.neg_type == "component_swap") & (df.ratio == 1) &
             (df.backbone == "distilbert-base-uncased")]
          .sort_values("strategy"))
    smap = {"chain_of_thought": "CoT", "few_shot": "Few-shot", "zero_shot": "Zero-shot"}
    rows_a3 = [base_row] + [(smap[r.strategy], r.acc_at_1, r.acc_at_5, r.mrr)
                             for _, r in a3.iterrows()]
    cot = a3[a3.strategy == "chain_of_thought"].iloc[0]
    _add_bar_panel(fig, rows_a3, 1, 1, show_legend=True,
                   annotations=[(f"Only CoT improves<br>Acc@1 above baseline<br><b>{cot.acc_at_1:.3f}</b>",
                                  0.73, cot.acc_at_1, 0.73, 0.72, BC1)],
                   suppress_text_at={"Acc@1": ["CoT"]},
                   c1=BC1, c5=BC5, cmrr=BCMRR, label_sz=16, annot_sz=17)

    # ── B. A4 Negative Type ───────────────────────────────────────────────────
    a4 = (df[(df.strategy == "chain_of_thought") & (df.ratio == 1) &
             (df.backbone == "distilbert-base-uncased")]
          .sort_values("neg_type"))
    tmap = {"abbreviation": "Abbrev.", "component_swap": "Comp. Swap",
            "phonetic": "Phonetic", "semantic_distractor": "Sem. Dist."}
    rows_a4 = [base_row] + [(tmap[r.neg_type], r.acc_at_1, r.acc_at_5, r.mrr)
                             for _, r in a4.iterrows()]
    cs = a4[a4.neg_type == "component_swap"].iloc[0]
    _add_bar_panel(fig, rows_a4, 1, 2, show_legend=False,
                   annotations=[(f"Comp. Swap best<br>on Acc@1 & MRR<br><b>{cs.acc_at_1:.3f}</b>",
                                  1.73, cs.acc_at_1, 1.73, 0.72, BC1)],
                   suppress_text_at={"Acc@1": ["Comp. Swap"]},
                   c1=BC1, c5=BC5, cmrr=BCMRR, label_sz=15, annot_sz=17)

    # ── C. A5 LLM-gen:in-batch Ratio ─────────────────────────────────────────
    # Explicit seed=42 + eval_split=val: the CSV now contains multi-seed and
    # test-set rows for ratio=4; without this filter .iloc[0] would be fragile.
    a5 = (df[(df.neg_type == "component_swap") & (df.strategy == "chain_of_thought") &
             (df.backbone == "distilbert-base-uncased") &
             (df.seed == 42) & (df.eval_split == "val")]
          .sort_values("ratio"))
    rmap = {0: "1:0 (LLM only)", 1: "1:1", 4: "4:1"}
    rows_a5 = [base_row] + [(rmap[r.ratio], r.acc_at_1, r.acc_at_5, r.mrr)
                             for _, r in a5.iterrows()]
    r4 = a5[a5.ratio == 4].iloc[0]
    _add_bar_panel(fig, rows_a5, 2, 1, show_legend=False,
                   annotations=[(f"4:1 boosts Acc@5<br>& MRR<br><b>{r4.acc_at_5:.3f}</b>",
                                  3, r4.acc_at_5, 3, 0.70, BC5)],
                   suppress_text_at={"Acc@5": ["4:1"]},
                   c1=BC1, c5=BC5, cmrr=BCMRR, label_sz=16, annot_sz=17)

    # ── D. A1 Backbone Sensitivity ────────────────────────────────────────────
    a1_df = (df[(df.neg_type == "component_swap") & (df.strategy == "chain_of_thought") &
                (df.ratio == 1)]
             .sort_values("backbone"))
    bmap = {"bert-base-uncased": "BERT base",
            "distilbert-base-uncased": "DistilBERT",
            "msmarco-distilbert-base-v4": "msmarco"}
    rows_a1 = [base_row] + [(bmap[r.backbone], r.acc_at_1, r.acc_at_5, r.mrr)
                             for _, r in a1_df.iterrows()]
    ms = a1_df[a1_df.backbone == "msmarco-distilbert-base-v4"].iloc[0]
    _add_bar_panel(fig, rows_a1, 2, 2, show_legend=False,
                   annotations=[(f"msmarco leads<br>Acc@5 & MRR<br><b>{ms.acc_at_5:.3f}</b>",
                                  3, ms.acc_at_5, 3, 0.74, BC5)],
                   suppress_text_at={"Acc@5": ["msmarco"]},
                   c1=BC1, c5=BC5, cmrr=BCMRR, label_sz=16, annot_sz=17)

    fig.update_layout(
        title=dict(
            text=(
                "<b>Figure 3: Breaking the Ceiling — Five-Factor Ablation Results</b><br>"
                f"train_10pct (698 pairs) | "
                f"Baseline: Acc@1={BASE_A1}, Acc@5={BASE_A5}, MRR={BASE_MRR}"
            ),
            font=dict(size=27, family=FONT, color="#111111"),
            x=0.5, xanchor="center",
            pad=dict(b=20),
        ),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.04, yanchor="bottom",
            font=dict(size=21, family=FONT),
            bgcolor="white", bordercolor="#cccccc", borderwidth=1,
        ),
        uniformtext=dict(minsize=15, mode='show'),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=14),
        margin=dict(l=80, r=80, t=280, b=70),
    )

    _save(fig, "fig3_ablation.pdf", w=1600, h=1260)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Data Efficiency Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig4() -> None:
    df      = pd.read_csv(RESULTS_DIR / "m3_results.csv")
    base_df = (pd.read_csv(RESULTS_DIR / "baselines.csv")
               .query("method == 'bi_encoder_in_batch'"))

    b10 = base_df[base_df.train_split == "train_10pct"].iloc[0]
    b25 = base_df[base_df.train_split == "train_25pct"].iloc[0]
    # Explicit seed=42 + eval_split=val: the CSV now contains multi-seed and
    # test-set rows for ratio=4; without this filter .iloc[0] would be fragile.
    r4  = df[(df.neg_type == "component_swap") & (df.strategy == "chain_of_thought") &
             (df.backbone == "distilbert-base-uncased") & (df.ratio == 4) &
             (df.seed == 42) & (df.eval_split == "val")].iloc[0]

    systems = [
        ("In-batch (10% data)",     "#0B2D72",
         b10["acc_at_1"], b10["acc_at_5"], b10["mrr"]),
        ("In-batch (25% data)",     "#853953",
         b25["acc_at_1"], b25["acc_at_5"], b25["mrr"]),
        ("LLM-HN (10% data, 4:1)", "#778873",
         r4["acc_at_1"],  r4["acc_at_5"],  r4["mrr"]),
    ]
    metrics = ["Acc@1", "Acc@5", "MRR"]

    fig = go.Figure()

    # In-batch 25% is the middle bar (index 1 of 3) in the "Acc@5" group (category index 1).
    # With bargap=0.2 and 3 bars, bar_width ≈ 0.8/3 ≈ 0.267; middle bar offset = 0.
    # So its center is at numeric x = 1 (the category index of "Acc@5").
    b25_acc5_x = 1

    for i, (label, color, v1, v5, vmrr) in enumerate(systems):
        vals = [v1, v5, vmrr]
        texts = [f"<b>{v:.4f}</b>" for v in vals]
        if i == 1:          # In-batch 25%: Acc@5 label suppressed (value shown in annotation)
            texts[1] = ""
        fig.add_trace(go.Bar(
            name=label, x=metrics, y=vals,
            marker=dict(color=color, opacity=0.88,
                        line=dict(color="white", width=0.5)),
            text=texts,
            textposition="outside",
            textfont=dict(size=15, family=FONT),
            cliponaxis=False,
        ))

    # Arrow annotation — straight down to In-batch 25% Acc@5 bar, value inside box
    fig.add_annotation(
        x=b25_acc5_x, y=b25["acc_at_5"],
        xref="x", yref="y",
        text=(f"<b>Acc@5 approaching</b><br>the in-batch scaling<br>"
              f"(+2.5× more data)<br><b>{b25['acc_at_5']:.4f}</b>"),
        showarrow=True, arrowhead=2, arrowcolor="#853953", arrowwidth=1.5,
        ax=b25_acc5_x, ay=0.74,
        axref="x", ayref="y",
        font=dict(size=15, color="#333333", family=FONT),
        bgcolor="white", bordercolor="#853953", borderwidth=1.5, borderpad=8,
        align="center",
    )

    # Note box — manual line break prevents right-side cut-off
    note = (
        f"★  <b>Note:</b> LLM-HN with 10% data (seed 42) reaches Acc@5 = {r4['acc_at_5']:.4f},"
        f" approaching the in-batch model trained with 25% data ({b25['acc_at_5']:.4f}).<br>"
        f"Mean across 3 seeds: 0.5506 ± 0.0032. Demonstrates strong data efficiency"
        f" of targeted hard negative fine-tuning."
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.29,
        text=note,
        showarrow=False,
        font=dict(size=16, color="#333333", family=FONT),
        bgcolor="#FAF7F0", bordercolor="#D8D2C2", borderwidth=1.5, borderpad=10,
        align="center",
    )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Figure 4: Data Efficiency Comparison — LLM-HN vs. In-batch Scaling</b>",
            font=dict(size=27, family=FONT, color="#111111"),
            x=0.5, xanchor="center",
        ),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.08, yanchor="bottom",
            font=dict(size=18, family=FONT),
            bgcolor="white", bordercolor="#cccccc", borderwidth=1,
        ),
        yaxis=dict(title_text="Score", range=[0, 0.85], tickformat=".2f",
                   **_yax(tickfont=dict(family=FONT, size=15),
                          title_font=dict(family=FONT, size=15))),
        xaxis=dict(title_text="",
                   **_xax(tickfont=dict(family=FONT, size=15),
                          title_font=dict(family=FONT, size=15))),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=15),
        margin=dict(l=70, r=70, t=190, b=140),
    )

    _save(fig, "fig4_data_efficiency.pdf", w=1000, h=660)


# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_fig1()
    plot_fig3()
    plot_fig4()
    print("All figures saved.")


if __name__ == "__main__":
    main()
