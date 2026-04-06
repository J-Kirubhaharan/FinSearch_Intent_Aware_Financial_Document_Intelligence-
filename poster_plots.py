"""
FinSearch — All Experiments Side-by-Side Comparison Plots
Run from project root: python poster_plots.py
Saves all images to poster_images/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR = 'visualization'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
GRAY   = '#95A5A6'; BLUE  = '#2980B9'; TEAL = '#1ABC9C'
GREEN  = '#27AE60'; GOLD  = '#F39C12'; PURP = '#8E44AD'
RED    = '#E74C3C'; DARK  = '#2C3E50'; LIGHT_BG = '#F8F9FA'

def bar_labels(ax, bars, fmt='{:.4f}', offset=0.006, fs=8.5):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + offset,
                fmt.format(b.get_height()), ha='center', fontsize=fs,
                fontweight='bold', color=DARK)

def clean_ax(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', path)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 1 — Master comparison: ALL retrieval experiments side by side
# ══════════════════════════════════════════════════════════════════════════════
def plot_master_comparison():
    """
    Single figure with 4 panels:
      Row 1: (A) Retrieval stages  |  (B) Full pipelines
      Row 2: (C) Chunking          |  (D) Intent classifier
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs  = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38)

    # ── Panel A: Retrieval Stages (FiQA 648 queries) ──────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    exp_a   = ['BM25\nBaseline', 'MiniLM\nDense', 'Hybrid\nRRF', 'Hybrid\nα=0.7', 'BGE-Large\n+ Rerank']
    ndcg_a  = [0.2169,           0.3687,           0.3519,         0.3791,           0.3885]
    mrr_a   = [0.2706,           0.4451,           0.4171,         0.4606,           0.4775]
    rec_a   = [0.2784,           0.4413,           0.4396,         0.4473,           0.4485]
    cols_a  = [GRAY, BLUE, TEAL, TEAL, GREEN]

    x_a = np.arange(len(exp_a)); w = 0.26
    b1 = ax_a.bar(x_a - w,   ndcg_a, w, color=cols_a, edgecolor='white', lw=1.2, label='NDCG@10')
    b2 = ax_a.bar(x_a,       mrr_a,  w, color=cols_a, edgecolor='white', lw=1.2, alpha=0.70, label='MRR')
    b3 = ax_a.bar(x_a + w,   rec_a,  w, color=cols_a, edgecolor='white', lw=1.2, alpha=0.45, label='Recall@10')
    for bars in [b1, b2, b3]:
        bar_labels(ax_a, bars, offset=0.005, fs=7.5)

    ax_a.set_xticks(x_a); ax_a.set_xticklabels(exp_a, fontsize=9)
    ax_a.set_ylim(0, 0.62); ax_a.set_ylabel('Score', fontsize=10)
    ax_a.set_title('(A) Retrieval Experiments — FiQA (648 queries)', fontweight='bold', fontsize=11)
    handles = [mpatches.Patch(color=DARK, alpha=a, label=l)
               for a, l in [(1.0,'NDCG@10'),(0.70,'MRR'),(0.45,'Recall@10')]]
    ax_a.legend(handles=handles, fontsize=8, loc='upper left')
    # winner border
    ax_a.patches[4*3].set_edgecolor(GOLD); ax_a.patches[4*3].set_linewidth(2.5)  # B1 NDCG
    clean_ax(ax_a)

    # ── Panel B: Full Pipelines (QE + Rerank + LLM, 194 queries) ─────────────
    ax_b = fig.add_subplot(gs[0, 1])

    exp_b  = ['A1\nMiniLM Dense\n+QE+Mistral', 'A2\nMiniLM Hybrid\n+QE+Mistral',
              'B1\nBGE-Large Dense\n+QE+Mistral ★', 'B2\nBGE-Large Hybrid\n+QE+Mistral']
    ndcg_b = [0.5917, 0.5813, 0.6056, 0.5381]
    mrr_b  = [0.6607, 0.6685, 0.6679, 0.6243]
    rec_b  = [0.6724, 0.6513, 0.6917, 0.5984]
    cols_b = [BLUE, TEAL, GREEN, PURP]

    x_b = np.arange(len(exp_b))
    c1 = ax_b.bar(x_b - w,  ndcg_b, w, color=cols_b, edgecolor='white', lw=1.2)
    c2 = ax_b.bar(x_b,      mrr_b,  w, color=cols_b, edgecolor='white', lw=1.2, alpha=0.70)
    c3 = ax_b.bar(x_b + w,  rec_b,  w, color=cols_b, edgecolor='white', lw=1.2, alpha=0.45)
    for bars in [c1, c2, c3]:
        bar_labels(ax_b, bars, offset=0.004, fs=7.5)

    ax_b.set_xticks(x_b); ax_b.set_xticklabels(exp_b, fontsize=8.5)
    ax_b.set_ylim(0, 0.85); ax_b.set_ylabel('Score', fontsize=10)
    ax_b.set_title('(B) Full Pipeline Comparison — QE + Rerank + LLM (194 queries)', fontweight='bold', fontsize=11)
    ax_b.legend(handles=handles, fontsize=8, loc='lower right')
    clean_ax(ax_b)

    # ── Panel C: Chunking Strategies ──────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])

    strat_labels = ['S1\nSliding\n512w', 'S2\nSliding\n256w', 'S3\nParagraph',
                    'S4\nToken-Exact\n(Winner ★)', 'S5\nSection\n200/400']
    rec_mini = [0.35, 0.40, 0.30, 0.75, 0.60]
    rec_bge  = [0.35, 0.40, 0.35, 0.80, 0.75]
    cols_c   = [GRAY if v < 0.65 else GREEN for v in rec_bge]

    x_c = np.arange(len(strat_labels))
    d1 = ax_c.bar(x_c - w/2, rec_mini, w*1.3, color=BLUE,  edgecolor='white', lw=1.2, label='MiniLM', alpha=0.85)
    d2 = ax_c.bar(x_c + w/2, rec_bge,  w*1.3, color=GREEN, edgecolor='white', lw=1.2, label='BGE-Large', alpha=0.85)
    bar_labels(ax_c, d1, fmt='{:.2f}', offset=0.01, fs=8.5)
    bar_labels(ax_c, d2, fmt='{:.2f}', offset=0.01, fs=8.5)

    ax_c.set_xticks(x_c); ax_c.set_xticklabels(strat_labels, fontsize=9)
    ax_c.set_ylim(0, 1.08); ax_c.set_ylabel('Recall@10', fontsize=10)
    ax_c.set_title('(C) Chunking Strategy Comparison — Recall@10 (4 PDFs, 20 QA pairs)', fontweight='bold', fontsize=11)
    ax_c.legend(fontsize=9)
    ax_c.axhline(0.70, color=GOLD, linestyle=':', alpha=0.6, lw=1.5)
    ax_c.text(4.7, 0.71, '0.70', fontsize=8, color=GOLD)
    clean_ax(ax_c)

    # ── Panel D: Intent Classifier ─────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    clf_labels = ['Zero-Shot\nDeBERTa NLI', 'MiniLM\n(PDF-Only\n600 q)', 'MiniLM\n(Full Dataset\nB77+PDF) ★']
    b77_acc = [25.5, 73.5, 93.0]
    qa_acc  = [5.0,  75.8, 90.0]
    cols_d  = [GRAY, BLUE, GREEN]

    x_d = np.arange(len(clf_labels))
    e1 = ax_d.bar(x_d - w/2, b77_acc, w*1.3, color=cols_d, edgecolor='white', lw=1.2, label='Banking77 Accuracy (%)')
    e2 = ax_d.bar(x_d + w/2, qa_acc,  w*1.3, color=cols_d, edgecolor='white', lw=1.2, alpha=0.70, label='QA Eval Accuracy (%)')
    bar_labels(ax_d, e1, fmt='{:.1f}%', offset=0.8, fs=9)
    bar_labels(ax_d, e2, fmt='{:.1f}%', offset=0.8, fs=9)

    ax_d.set_xticks(x_d); ax_d.set_xticklabels(clf_labels, fontsize=9)
    ax_d.set_ylim(0, 115); ax_d.set_ylabel('Accuracy (%)', fontsize=10)
    ax_d.set_title('(D) Intent Classifier — 4-Category Financial Routing\n(Banking77 + 120-q PDF eval)', fontweight='bold', fontsize=11)
    ax_d.legend(fontsize=9)
    ax_d.axhline(80, color=GOLD, linestyle='--', alpha=0.6, lw=1.5)
    ax_d.text(2.55, 81, '80% target', fontsize=8, color=GOLD)
    clean_ax(ax_d)

    fig.suptitle('FinSearch: Intent-Aware Financial Document Intelligence — All Experiments',
                 fontsize=16, fontweight='bold', color=DARK, y=1.01)

    save(fig, '01_all_experiments_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 2 — Clean metric table as image (for slides)
# ══════════════════════════════════════════════════════════════════════════════
def plot_metrics_table():
    fig, ax = plt.subplots(figsize=(15, 7), facecolor='white')
    ax.axis('off')

    # All rows: Experiment | Method | NDCG@10 | MRR | Recall@10 | Notes
    headers = ['Week', 'Experiment', 'Method / Config', 'NDCG@10', 'MRR', 'Recall@10', 'Notes']

    rows = [
        # Week 1 — Baseline
        ['Wk 1', 'Baseline', 'BM25 (k1=1.2, b=0.75)',        '0.2169', '0.2706', '0.2784', '648 queries'],
        # Week 2 — Dense
        ['Wk 2', 'Dense', 'MiniLM-L6-v2 FAISS',              '0.3687', '0.4451', '0.4413', '648 queries'],
        # Week 3 — Hybrid
        ['Wk 3', 'Hybrid', 'BM25 + MiniLM RRF (k=60)',       '0.3519', '0.4171', '0.4396', '648 queries'],
        ['Wk 3', 'Hybrid', 'BM25 + MiniLM α=0.7 ✓',         '0.3791', '0.4606', '0.4473', '648 queries'],
        # Week 4 — Rerank + Pipeline
        ['Wk 4', 'Rerank', 'Hybrid α=0.7 + Mistral Rerank',  '0.3885', '0.4775', '0.4485', '648 queries'],
        ['Wk 4', 'Pipeline A1', 'MiniLM Dense+QE+Mistral',   '0.5917', '0.6607', '0.6724', '194 queries'],
        ['Wk 4', 'Pipeline A2', 'MiniLM Hybrid+QE+Mistral',  '0.5813', '0.6685', '0.6513', '194 queries'],
        ['Wk 4', 'Pipeline B1 ★', 'BGE-L Dense+QE+Mistral',  '0.6056', '0.6679', '0.6917', '194 queries'],
        ['Wk 4', 'Pipeline B2', 'BGE-L Hybrid+QE+Mistral',   '0.5381', '0.6243', '0.5984', '194 queries'],
    ]

    # Add chunking rows (different metric — Recall@10 only)
    chunk_rows = [
        ['Wk 0', 'Chunking S1', 'Sliding 512w — MiniLM',     '—',      '—',      '0.35',   '20 QA pairs'],
        ['Wk 0', 'Chunking S2', 'Sliding 256w — MiniLM',     '—',      '—',      '0.40',   '20 QA pairs'],
        ['Wk 0', 'Chunking S3', 'Paragraph — MiniLM',        '—',      '—',      '0.30',   '20 QA pairs'],
        ['Wk 0', 'Chunking S4 ★','Token-Exact 200/400',      '—',      '—',      '0.80',   '20 QA pairs'],
        ['Wk 0', 'Chunking S5', 'Section 200/400',           '—',      '—',      '0.75',   '20 QA pairs'],
    ]

    intent_rows = [
        ['Wk 5', 'Intent Clf', 'Zero-Shot DeBERTa NLI',      '—', 'B77: 25.5%', 'QA: 5.0%',  '4 categories'],
        ['Wk 5', 'Intent Clf', 'MiniLM PDF-Only (600 q)',    '—', 'B77: 73.5%', 'QA: 75.8%', '120 held-out'],
        ['Wk 5', 'Intent Clf ★','MiniLM Full (B77+PDF)',     '—', 'B77: 93.0%', 'QA: 90.0%', '120 held-out'],
    ]

    all_rows = rows + chunk_rows + intent_rows

    # Colour-code winner rows
    row_colors = []
    for r in all_rows:
        if '★' in r[1]:
            row_colors.append(['#D5F5E3'] * len(headers))
        elif r[0] == 'Wk 4':
            row_colors.append(['#EAF4FB'] * len(headers))
        elif r[0] == 'Wk 0':
            row_colors.append(['#FEF9E7'] * len(headers))
        elif r[0] == 'Wk 5':
            row_colors.append(['#F4ECF7'] * len(headers))
        else:
            row_colors.append(['#FDFEFE'] * len(headers))

    table = ax.table(
        cellText=all_rows,
        colLabels=headers,
        cellColours=row_colors,
        colColours=[DARK] * len(headers),
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    # Header text white
    for j in range(len(headers)):
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('FinSearch — All Experiment Results at a Glance',
                 fontsize=15, fontweight='bold', color=DARK, pad=18)

    # Legend
    legend_items = [
        mpatches.Patch(color='#D5F5E3', label='★ Winner / Best config'),
        mpatches.Patch(color='#EAF4FB', label='Week 4 — Full pipeline'),
        mpatches.Patch(color='#FEF9E7', label='Week 0 — Chunking'),
        mpatches.Patch(color='#F4ECF7', label='Week 5 — Intent classification'),
    ]
    ax.legend(handles=legend_items, loc='lower center', fontsize=9,
              ncol=4, bbox_to_anchor=(0.5, -0.04))

    save(fig, '02_metrics_table.png')


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 3 — Progressive NDCG@10 improvement bar (clean, poster-sized)
# ══════════════════════════════════════════════════════════════════════════════
def plot_ndcg_progression():
    stages = ['BM25\nBaseline', 'MiniLM\nDense', 'Hybrid\nα=0.7', 'Hybrid+\nRerank', 'Full Pipeline\nB1 (sub-corpus)']
    ndcg   = [0.2169,           0.3687,            0.3791,          0.3885,            0.6056]
    colors = [GRAY, BLUE, TEAL, GREEN, GOLD]
    deltas = [None] + [ndcg[i]-ndcg[i-1] for i in range(1, len(ndcg))]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor='white')
    bars = ax.bar(stages, ndcg, color=colors, width=0.55, edgecolor='white', linewidth=2)

    for bar, val, delta in zip(bars, ndcg, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{val:.4f}', ha='center', fontsize=12, fontweight='bold', color=DARK)
        if delta:
            sign = '+' if delta > 0 else ''
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                    f'{sign}{delta:.4f}', ha='center', fontsize=9.5,
                    color=GREEN if delta > 0 else RED, fontweight='bold')

    # dashed trend line
    xs = [b.get_x() + b.get_width()/2 for b in bars]
    ax.plot(xs, ndcg, color=DARK, lw=1.5, linestyle='--', alpha=0.35, zorder=5)

    ax.set_ylim(0, 0.75)
    ax.set_ylabel('NDCG@10', fontsize=12)
    ax.set_title('NDCG@10 Improvement Across All Stages — FinSearch Retrieval Pipeline',
                 fontsize=13, fontweight='bold', color=DARK, pad=14)
    clean_ax(ax)
    ax.tick_params(axis='x', labelsize=11)

    note = '* B1 sub-corpus uses stratified 30% of FiQA + all qrel docs (194 queries)'
    ax.text(0.5, -0.14, note, transform=ax.transAxes, ha='center', fontsize=9,
            color='gray', style='italic')

    save(fig, '03_ndcg_progression.png')


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 4 — Recall@10 across ALL experiments (chunking + retrieval + pipeline)
# ══════════════════════════════════════════════════════════════════════════════
def plot_recall_all():
    labels = [
        'BM25\nBaseline',
        'MiniLM\nDense',
        'Hybrid\nα=0.7',
        'BGE+\nRerank',
        '─',
        'Chunk S1\nSlide-512',
        'Chunk S2\nSlide-256',
        'Chunk S3\nParagraph',
        'Chunk S4\nToken-Exact\n(★)',
        'Chunk S5\nSection',
        '─',
        'Pipeline\nA1',
        'Pipeline\nA2',
        'Pipeline\nB1 (★)',
        'Pipeline\nB2',
    ]
    recall = [0.2784, 0.4413, 0.4473, 0.4485,
              None,
              0.35, 0.40, 0.30, 0.80, 0.75,
              None,
              0.6724, 0.6513, 0.6917, 0.5984]
    colors = [GRAY, BLUE, TEAL, GREEN,
              'white',
              GRAY, BLUE, RED, GREEN, TEAL,
              'white',
              BLUE, TEAL, GREEN, PURP]

    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor='white')

    x = np.arange(len(labels))
    for xi, (val, col, lbl) in enumerate(zip(recall, colors, labels)):
        if val is None:
            continue
        bar = ax.bar(xi, val, color=col, width=0.65, edgecolor='white', linewidth=1.5)
        ax.text(xi, val + 0.012, f'{val:.2f}', ha='center', fontsize=8.5,
                fontweight='bold', color=DARK)

    # Separator lines
    ax.axvline(4,  color='#BDC3C7', lw=1.5, linestyle='--')
    ax.axvline(10, color='#BDC3C7', lw=1.5, linestyle='--')
    ax.text(2.0, 0.92, 'Full Corpus\n(648 q)', ha='center', fontsize=9, color='gray')
    ax.text(7.5, 0.92, 'PDF Chunking\n(20 QA pairs)', ha='center', fontsize=9, color='gray')
    ax.text(12.5, 0.92, 'Full Pipeline\n(194 q)', ha='center', fontsize=9, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Recall@10 — All Experiments Side by Side',
                 fontsize=13, fontweight='bold', color=DARK, pad=14)
    clean_ax(ax)

    save(fig, '04_recall_all_experiments.png')


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating poster images...\n')
    plot_master_comparison()
    plot_metrics_table()
    plot_ndcg_progression()
    plot_recall_all()
    print(f'\nAll images saved to: {OUT_DIR}/')
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f'  {f}  ({size} KB)')
