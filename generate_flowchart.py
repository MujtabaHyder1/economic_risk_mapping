"""
End-to-End Pipeline Architecture Flowchart Generator
Generates a professional architectural flowchart saved as:
C:\\Users\\Mujtaba\\Desktop\\Final\\ML\\economic_risk_mapping\\results\\pipeline_architecture_flowchart.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# Set up figure with publication quality
fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color Palette (Kashmir Blue / Emerald Tech Theme)
C_BG = '#0F172A'       # Deep Slate Background
C_BOX_DATA = '#1E293B' # Slate Blue Box
C_BOX_PROC = '#3B82F6' # Blue Process Box
C_BOX_ML = '#10B981'   # Emerald ML Box
C_BOX_OUT = '#8B5CF6'  # Purple Output Box
C_TEXT = '#F1F5F9'     # Near White Text
C_MUTED = '#94A3B8'    # Muted Gray Text
C_ACCENT = '#38BDF8'   # Cyan Accent

# Set figure background
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

# Title Block
ax.text(50, 95, "ECONOMIC RISK MAPPING — END-TO-END PIPELINE ARCHITECTURE",
        fontsize=16, fontweight='bold', color=C_TEXT, ha='center', va='center', family='sans-serif')
ax.text(50, 91, "From Multi-Source Synthetic Data Generation to Climate Scenario Vulnerability Indexing",
        fontsize=11, color=C_ACCENT, ha='center', va='center', family='sans-serif')

def draw_box(x, y, w, h, title, subtitle, color):
    # Rounded box
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5",
                                  facecolor=color, edgecolor='#64748B', linewidth=1.5, alpha=0.95)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.6, title, fontsize=11, fontweight='bold', color='#FFFFFF', ha='center', va='center')
    ax.text(x + w/2, y + h/2 - 0.7, subtitle, fontsize=8.5, color='#E2E8F0', ha='center', va='center')

def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color='#38BDF8', lw=2, mutation_scale=12, connectionstyle="arc3"))
    if label:
        ax.mid = ((x1+x2)/2, (y1+y2)/2)
        ax.text(ax.mid[0] + 0.8, ax.mid[1], label, fontsize=8, color='#94A3B8', ha='left', va='center')

# --- PHASE 1: DATA GENERATION (Top Row) ---
ax.text(12, 83, "PHASE 1: SYNTHETIC DATA GENERATION", fontsize=11, fontweight='bold', color=C_ACCENT, ha='center')
draw_box(3, 70, 18, 9, "Climate Data", "Daily Temp & Precip\n(14,247 records)", C_BOX_DATA)
draw_box(23, 70, 18, 9, "Geospatial Data", "Monthly NDVI & Altitude\n(468 records)", C_BOX_DATA)
draw_box(43, 70, 18, 9, "Economic Data", "Yearly Yield & Prices\n(117 records)", C_BOX_DATA)

# --- PHASE 2: PREPROCESSING & FEATURE ENGINEERING (Middle-Top Row) ---
ax.text(35, 57, "PHASE 2: PREPROCESSING & FEATURE ENGINEERING", fontsize=11, fontweight='bold', color=C_ACCENT, ha='center')
draw_box(15, 45, 36, 9, "Data Merging & Cleaning", "Multi-key join on (Year, District, Crop)\nForward-fill imputation (18 nulls) | IQR Outlier Removal (9 rows dropped)", C_BOX_PROC)
draw_box(55, 45, 30, 9, "Feature Engineering (16 Predictors)", "Frost_Days_March, Hot_Days_JulAug\nNDVI_Deviation, Altitude_Mean, Annual Temp Metrics", C_BOX_PROC)

# Arrows from Phase 1 to Phase 2
draw_arrow(12, 70, 25, 54)
draw_arrow(32, 70, 30, 54)
draw_arrow(52, 70, 35, 54)
draw_arrow(33, 45, 55, 49) # Internal merge to feature eng

# --- PHASE 3: MODEL TRAINING & TUNING (Middle-Bottom Row) ---
ax.text(35, 32, "PHASE 3: MACHINE LEARNING MODEL TRAINING", fontsize=11, fontweight='bold', color=C_ACCENT, ha='center')
draw_box(8, 20, 26, 9, "Temporal Train-Test Split", "Train: 2012–2023 (99 samples)\nTest: 2024 (9 samples — No Data Leakage)", C_BOX_ML)
draw_box(37, 20, 26, 9, "GridSearchCV Tuning", "5-Fold CV across 216 combinations\n(1,080 total fits)", C_BOX_ML)
draw_box(66, 20, 26, 9, "XGBoost Regressor", "Optimized: depth=4, n_est=300, lr=0.05\nTest R²: 0.9772 | RMSE: ₹62,994/ha", C_BOX_ML)

# Arrows in Phase 3
draw_arrow(50, 45, 21, 29)
draw_arrow(34, 24, 37, 24)
draw_arrow(63, 24, 66, 24)

# --- PHASE 4: EVALUATION & VULNERABILITY INDEX (Bottom Row) ---
ax.text(78, 57, "PHASE 4: VULNERABILITY & DECISION INTELLIGENCE", fontsize=11, fontweight='bold', color=C_ACCENT, ha='center')
draw_box(60, 45, 36, 9, "Climate Scenario Simulation", "Simulating +1.5°C and +2.0°C Warming Scenarios\nCalculating Percentage Revenue Loss per District-Crop", C_BOX_OUT)

draw_box(60, 5, 36, 11, "Financial Vulnerability Index (FVI)", "Percentile Normalization (0–100 Scale)\n• Shopian Saffron: 100.0 Critical Risk (-29.7% loss)\n• Baramulla: 40.5 High Risk | Pulwama: 18.8 Low Risk\n• Actionable Policy: Targeted Insurance & Frost Infrastructure", C_BOX_OUT)

# Arrows to Phase 4
draw_arrow(92, 29, 78, 45)
draw_arrow(78, 45, 78, 16)

plt.tight_layout()
output_path = r"C:\Users\Mujtaba\Desktop\Final\ML\economic_risk_mapping\results\pipeline_architecture_flowchart.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Flowchart successfully generated at: {output_path}")
