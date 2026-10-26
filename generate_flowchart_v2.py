"""
Improved flowchart with better spacing and readability.
Vertical phased layout.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 14), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A')

C_TEXT = '#F1F5F9'
C_ACCENT = '#10B981'
C_BOX_DATA = '#1E293B'
C_BOX_PROC = '#3B82F6'
C_BOX_MODEL = '#10B981'
C_BOX_OUTPUT = '#8B5CF6'
C_ARROW = '#38BDF8'

def box(x_center, y_bottom, width, height, title, subtitle, color):
    x0 = x_center - width/2
    y0 = y_bottom
    box = patches.FancyBboxPatch((x0, y0), width, height,
                                boxstyle="round,pad=0.2,rounding_size=0.3",
                                facecolor=color, edgecolor='#64748B', linewidth=1.2)
    ax.add_patch(box)
    ax.text(x_center, y0 + height*0.65, title,
            fontsize=10, fontweight='bold', color='white', ha='center', va='center')
    ax.text(x_center, y0 + height*0.35, subtitle,
            fontsize=8, color='#E2E8F0', ha='center', va='center')

def arrow_up(x, y1, y2, label=None):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5, mutation_scale=10))
    if label:
        ax.text(x+0.3, (y1+y2)/2, label, fontsize=8, color=C_ACCENT, va='center', ha='left')

# Title
ax.text(5, 13.5, "ECONOMIC RISK MAPPING PIPELINE", fontsize=14, fontweight='bold', color=C_TEXT, ha='center')
ax.text(5, 12.8, "From Data Generation → Feature Eng → Modeling → Vulnerability Index", fontsize=9, color=C_ACCENT, ha='center')

# Phase 1 Boxes (three side-by-side)
box(3, 11, 2.5, 1.2, "Climate Data\n(Daily)", "14,247 records (2012-2024)", C_BOX_DATA)
box(5, 11, 2.5, 1.2, "Geospatial Data\n(Monthly)", "468 NDVI & Altitude records", C_BOX_DATA)
box(7, 11, 2.5, 1.2, "Economic Data\n(Yearly)", "117 Farm‑gate/Yield records", C_BOX_DATA)

# Arrow down from middle of phase1 cluster
arrow_up(5, 11, 9.5, "")

# Phase 2: Preprocessing & Feature Eng (single wide box)
box(5, 8, 6, 1.5, "Preprocessing & Feature Engineering",
     "Temporal agg · Forward‑fill impute (18 nulls) · IQR Outlier Removal (9 rows)\nEngineered 16 features: Frost_Days_March, Hot_Days_JulAug, NDVI_Deviation, Altitude_Mean, Temp metrics", C_BOX_PROC)
arrow_up(5, 8, 6.5, "")  # from bottom of proc to top of model?

# Phase 3: Model Training (two boxes side by side: split and training)
box(3, 5, 3, 1.2, "Temporal Train‑Test Split",
     "Train: 2012‑2023 (99 samples)\nTest: 2024 (9 samples) – Zero leakage", C_BOX_MODEL)
box(7, 5, 3, 1.2, "XGBoost + GridSearchCV",
     "216 combos · 5‑Fold CV (1,080 fits)\nBest: depth=4, n_est=300, lr=0.05, subsample=0.7, colsample=0.85", C_BOX_MODEL)
# Arrow connecting split to training (horizontal)
ax.annotate("", xy=(6, 5.6), xytext=(4.5, 5.6),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.2))
ax.text(5.25, 5.8, "Prepared Features", fontsize=7, color=C_ACCENT, ha='center')
arrow_up(5, 5, 4, "")  # from model phase down to output

# Phase 4: Evaluation & Vulnerability Index
box(5, 2, 6, 2, "Evaluation & Financial Vulnerability Index",
     "• Metrics: R²=0.9772, RMSE=₹62,994/ha, MAE=₹47,630, MAPE=18.95%\n• Feature Importance bar chart\n• +1.5°C & +2.0°C scenarios → Vulnerability Index (0‑100)\n• Actionable insights for policy & frost protection", C_BOX_OUTPUT)

# Optionally caption
ax.text(5, 0.5, "End‑to‑end pipeline: single command `python main.py` → reproducible results", fontsize=8, color=C_ACCENT, ha='center')

plt.tight_layout()
out_path = r"C:\Users\Mujtaba\Desktop\Final\ML\economic_risk_mapping\results\pipeline_architecture_flowchart_v2.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Improved flowchart saved to: {out_path}")
