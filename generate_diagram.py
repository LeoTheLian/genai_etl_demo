import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(18, 22))
ax.set_xlim(0, 18)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── colour palette ────────────────────────────────────────────────────────────
C_AGENT    = "#1e3a5f"   # deep blue  – agent boxes
C_FILE     = "#1a2e1a"   # deep green – file/artifact boxes
C_ORCH     = "#3a1f5e"   # purple     – orchestrator
C_EXEC     = "#1f3a2e"   # teal-green – executor
C_BORDER_A = "#4a9eff"   # blue border
C_BORDER_F = "#4caf50"   # green border
C_BORDER_O = "#9c6eff"   # purple border
C_BORDER_E = "#26c6a0"   # teal border
C_TEXT     = "#e8eaf6"
C_FILE_TXT = "#c8e6c9"
C_ARROW    = "#78909c"
C_LABEL    = "#b0bec5"
C_FEEDBACK = "#ff7043"

def agent_box(ax, x, y, w, h, title, lines, bg, border, title_color=C_TEXT):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                         facecolor=bg, edgecolor=border, linewidth=2.0, zorder=3)
    ax.add_patch(box)
    # title
    ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top",
            fontsize=10, fontweight="bold", color=title_color, zorder=4)
    # divider line
    ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.52, y + h - 0.52],
            color=border, lw=0.8, alpha=0.6, zorder=4)
    # body lines
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.72 - i * 0.28, line, ha="center", va="top",
                fontsize=7.5, color=C_LABEL, zorder=4)

def file_box(ax, x, y, w, h, title, lines, border=C_BORDER_F):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                         facecolor=C_FILE, edgecolor=border, linewidth=1.4,
                         linestyle="dashed", zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top",
            fontsize=8, fontweight="bold", color=C_FILE_TXT, zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.50 - i * 0.24, line, ha="center", va="top",
                fontsize=7, color="#81c784", zorder=4)

def arrow(ax, x1, y1, x2, y2, label="", color=C_ARROW, lw=1.6,
          style="->", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.8,
                color=color, style="italic",
                bbox=dict(facecolor="#0f1117", edgecolor="none", pad=1.5),
                zorder=5)

# ── layout constants ──────────────────────────────────────────────────────────
CX, CW = 6.5, 5.0          # centre column x, width
FW_WIDE, FH = 4.6, 1.05    # wide file box
FW_NRW = 3.6                # narrow file box

# ── INPUT FILES (row 21) ──────────────────────────────────────────────────────
file_box(ax, 1.2, 20.0, FW_WIDE, 1.4,
         "requirements_document.txt",
         ["Plain-text pipeline spec", "Schema, renames, derivations"])
file_box(ax, 10.8, 20.0, FW_WIDE, 1.4,
         "sample_transactions.csv",
         ["Raw source data", "284,807 transactions"])

# ── ORCHESTRATOR (row ~18.3) ──────────────────────────────────────────────────
agent_box(ax, CX, 17.8, CW, 1.55,
          "ORCHESTRATOR",
          ["orchestrator.py  ·  entry point",
           "Coordinates all agents · manages retry loop",
           "Executes pipeline · logs activity"],
          C_ORCH, C_BORDER_O, title_color="#ce93d8")

# ── ANALYST AGENT (row ~15.4) ─────────────────────────────────────────────────
agent_box(ax, CX, 15.0, CW, 1.85,
          "ANALYST AGENT",
          ["analyst_agent.py",
           "Parses requirements document",
           "Profiles source CSV (stats, nulls, samples)",
           "LLM enriches data dictionary"],
          C_AGENT, C_BORDER_A)

# ── MAPPING + DICTIONARY outputs (row ~12.8) ─────────────────────────────────
file_box(ax, 2.5, 12.5, FW_WIDE, 1.4,
         "source_to_target_mapping.json",
         ["Column renames · type casts",
          "Filter rules · derivation logic"])
file_box(ax, 10.0, 12.5, FW_WIDE, 1.4,
         "source_data_dictionary.json",
         ["Column stats · null rates",
          "Business meanings (LLM)"])

# ── DEVELOPER AGENT (row ~10.2) ───────────────────────────────────────────────
agent_box(ax, CX, 10.0, CW, 1.85,
          "DEVELOPER AGENT",
          ["developer_agent.py",
           "Builds detailed LLM prompt",
           "LLM writes run() function body",
           "Wraps in executable pipeline template"],
          C_AGENT, C_BORDER_A)

# ── ETL UTILITIES (side, row ~10) ─────────────────────────────────────────────
file_box(ax, 13.2, 10.1, 4.3, 1.4,
         "etl_utilities.py",
         ["Reusable transform functions", "load/save · rename · cast · derive"],
         border="#f4a261")

# ── GENERATED PIPELINE (row ~8.2) ─────────────────────────────────────────────
file_box(ax, CX, 8.2, CW, 1.3,
         "generated_pipeline.py",
         ["Executable Python ETL script",
          "Calls etl_utilities functions"])

# ── EXECUTOR (row ~6.2) ───────────────────────────────────────────────────────
agent_box(ax, CX, 6.1, CW, 1.85,
          "PIPELINE EXECUTION",
          ["Run by Orchestrator as subprocess",
           "Applies all renames · casts · filters",
           "Derives analytics columns",
           "Writes output CSV + validation report"],
          C_EXEC, C_BORDER_E, title_color="#80cbc4")

# ── PROCESSED + REPORT outputs (row ~4.0) ────────────────────────────────────
file_box(ax, 2.5, 3.8, FW_WIDE, 1.4,
         "fraud_transactions.csv",
         ["Processed output dataset",
          "Renamed · typed · enriched rows"])
file_box(ax, 10.0, 3.8, FW_WIDE, 1.4,
         "validation_report.txt",
         ["Row counts in/out",
          "Rejection counts by reason"])

# ── TESTER AGENT (row ~1.8) ───────────────────────────────────────────────────
agent_box(ax, CX, 1.6, CW, 1.85,
          "TESTER AGENT",
          ["tester_agent.py",
           "LLM generates data-quality test cases",
           "Executes tests on processed output",
           "Returns structured pass/fail results"],
          C_AGENT, C_BORDER_A)

# ── TEST REPORT (row ~0.0) ────────────────────────────────────────────────────
file_box(ax, CX + 0.7, 0.05, FW_NRW, 1.1,
         "test_report.txt",
         ["Per-test pass/fail", "Data quality summary"])

# ── ARROWS ────────────────────────────────────────────────────────────────────
MID_C = CX + CW / 2   # 9.0  horizontal centre of agent column

# inputs → orchestrator
arrow(ax, 3.5,  20.0, MID_C - 1.5, 19.35, "requirements path")
arrow(ax, 13.1, 20.0, MID_C + 1.5, 19.35, "source CSV path")

# orchestrator → analyst
arrow(ax, MID_C, 17.8, MID_C, 16.85, "requirements + source CSV")

# analyst → mapping / dictionary
arrow(ax, MID_C - 0.5, 15.0, 4.8, 13.9,  "mapping JSON")
arrow(ax, MID_C + 0.5, 15.0, 12.3, 13.9, "dictionary JSON")

# mapping / dictionary → developer
arrow(ax, 4.8,  12.5, MID_C - 0.5, 11.85, "")
arrow(ax, 12.3, 12.5, MID_C + 0.5, 11.85, "")

# utilities → pipeline (dashed)
arrow(ax, 13.2, 10.8, CX + CW, 9.6, "imported at runtime",
      color="#f4a261", lw=1.2, style="-|>")

# developer → generated pipeline
arrow(ax, MID_C, 10.0, MID_C, 9.5, "")

# generated pipeline → executor
arrow(ax, MID_C, 8.2, MID_C, 7.95, "executed by orchestrator")

# executor → processed / report
arrow(ax, MID_C - 0.5, 6.1, 4.8, 5.2,  "processed CSV")
arrow(ax, MID_C + 0.5, 6.1, 12.3, 5.2, "validation report")

# processed → tester
arrow(ax, 4.8, 3.8, MID_C - 0.5, 3.45, "")

# orchestrator also sends req + source to tester (long left-side arrow)
arrow(ax, CX, 18.57, 0.8, 18.57, "", color=C_ARROW, lw=1.0)
arrow(ax, 0.8, 18.57, 0.8, 2.52, "", color=C_ARROW, lw=1.0)
arrow(ax, 0.8, 2.52, CX, 2.52, "req + source CSV to tester",
      color=C_ARROW, lw=1.0)

# tester → test report
arrow(ax, MID_C, 1.6, MID_C, 1.15, "")

# FEEDBACK LOOP: tester → orchestrator → developer (right side)
arrow(ax, CX + CW, 2.52,  17.4, 2.52,  "", color=C_FEEDBACK, lw=1.6)
arrow(ax, 17.4,    2.52,  17.4, 18.57, "", color=C_FEEDBACK, lw=1.6)
arrow(ax, 17.4,   18.57,  CX + CW, 18.57, "failed test results",
      color=C_FEEDBACK, lw=1.6)

arrow(ax, CX + CW, 18.57, 17.8, 18.57, "", color=C_FEEDBACK, lw=1.6)
arrow(ax, 17.8,    18.57, 17.8, 10.92, "", color=C_FEEDBACK, lw=1.6)
arrow(ax, 17.8,    10.92, CX + CW, 10.92,
      "test_feedback (on failure)", color=C_FEEDBACK, lw=1.6)

# ── LEGEND ────────────────────────────────────────────────────────────────────
lx, ly = 0.3, 6.5
ax.text(lx, ly + 1.6, "Legend", fontsize=8, fontweight="bold",
        color=C_TEXT, va="top")
for label, color, ls in [
    ("Agent",           C_BORDER_A, "solid"),
    ("Orchestrator",    C_BORDER_O, "solid"),
    ("Executor",        C_BORDER_E, "solid"),
    ("File / Artifact", C_BORDER_F, "dashed"),
]:
    rect = FancyBboxPatch((lx, ly + 0.05), 0.55, 0.3,
                          boxstyle="round,pad=0.04",
                          facecolor="#1a1a2e", edgecolor=color,
                          linewidth=1.5, linestyle=ls, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.7, ly + 0.22, label, fontsize=7, color=C_LABEL, va="center")
    ly -= 0.45

ax.plot([lx, lx + 0.55], [ly + 0.22, ly + 0.22], color=C_FEEDBACK, lw=1.8)
ax.annotate("", xy=(lx + 0.55, ly + 0.22), xytext=(lx + 0.4, ly + 0.22),
            arrowprops=dict(arrowstyle="->", color=C_FEEDBACK, lw=1.5), zorder=3)
ax.text(lx + 0.7, ly + 0.22, "Feedback loop", fontsize=7, color=C_LABEL, va="center")

# ── TITLE ─────────────────────────────────────────────────────────────────────
ax.text(9, 21.7, "GenAI ETL — Multi-Agent Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=C_TEXT)

plt.tight_layout(pad=0.3)
plt.savefig("docs/architecture_diagram.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved docs/architecture_diagram.png")
