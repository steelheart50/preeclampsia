# ============================================================
# FILE 2: train_model.py
# PURPOSE: Trains XGBoost model, runs SHAP explainability,
#          runs Fairlearn bias audit, saves all outputs
# HOW TO RUN: python train_model.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # prevents popup windows
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, brier_score_loss,
                             RocCurveDisplay)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from fairlearn.metrics import (MetricFrame, selection_rate,
                                false_positive_rate, false_negative_rate)
from sklearn.metrics import accuracy_score

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ═══════════════════════════════════════════════════════════
# PART 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 55)
print("PREECLAMPSIA RISK MODEL — FULL PIPELINE")
print("=" * 55)

df = pd.read_csv('data/preeclampsia_dataset.csv')
print(f"\n✅ Data loaded: {len(df)} patients, {df.shape[1]} columns")
print(f"   Preeclampsia rate: {df['preeclampsia'].mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════
# PART 2 — PREPARE FEATURES
# ═══════════════════════════════════════════════════════════

# Encode ethnicity as numbers (ML needs numbers, not text)
le = LabelEncoder()
df['ethnicity_encoded'] = le.fit_transform(df['ethnicity'])

FEATURES = ['age', 'bmi', 'ethnicity_encoded', 'nulliparous',
            'chronic_htn', 'diabetes', 'kidney_disease',
            'map', 'utapi', 'plgf', 'sflt1', 'uric_acid']

X = df[FEATURES]
y = df['preeclampsia']
ethnicity_col = df['ethnicity']  # keep for bias analysis

# ── Train / Test split (80% train, 20% test) ─────────────
X_train, X_test, y_train, y_test, eth_train, eth_test = train_test_split(
    X, y, ethnicity_col, test_size=0.2, random_state=42, stratify=y
)

print(f"\n   Training set: {len(X_train)} patients")
print(f"   Test set:     {len(X_test)} patients")

# ═══════════════════════════════════════════════════════════
# PART 3 — TRAIN XGBOOST MODEL
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("TRAINING XGBOOST MODEL...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handle imbalance
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# ── Cross-validation on training set ─────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"   5-fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── Test set predictions ─────────────────────────────────
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# ═══════════════════════════════════════════════════════════
# PART 4 — PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("MODEL PERFORMANCE (TEST SET)")
print("─" * 55)

auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"   AUROC:       {auc:.3f}")
print(f"   Brier Score: {brier:.3f}  (lower = better calibrated)")
print(f"   Sensitivity: {report['1']['recall']:.3f}")
print(f"   Specificity: {report['0']['recall']:.3f}")
print(f"   PPV:         {report['1']['precision']:.3f}")
print(f"   NPV:         {report['0']['precision']:.3f}")

# ── Save metrics to text file ─────────────────────────────
with open('outputs/model_metrics.txt', 'w') as f:
    f.write("PREECLAMPSIA MODEL — PERFORMANCE METRICS\n")
    f.write("=" * 45 + "\n")
    f.write(f"AUROC:       {auc:.3f}\n")
    f.write(f"Brier Score: {brier:.3f}\n")
    f.write(f"Sensitivity: {report['1']['recall']:.3f}\n")
    f.write(f"Specificity: {report['0']['recall']:.3f}\n")
    f.write(f"PPV:         {report['1']['precision']:.3f}\n")
    f.write(f"NPV:         {report['0']['precision']:.3f}\n")
    f.write(f"CV AUC:      {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")

# ── ROC Curve ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax,
                                  name=f'XGBoost (AUC={auc:.3f})')
ax.set_title('ROC Curve — Preeclampsia Prediction', fontsize=13, fontweight='bold')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=150)
plt.close()

# ── Calibration Plot ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
ax.plot(prob_pred, prob_true, 's-', color='steelblue', label='Model')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=11)
ax.set_ylabel('Fraction of Positives', fontsize=11)
ax.set_title('Calibration Plot', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/calibration_plot.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════
# PART 5 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("RUNNING SHAP ANALYSIS (this takes ~30 seconds)...")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# ── Global: Feature Importance Bar Plot ──────────────────
fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.bar(shap_values, show=False, ax=ax)
ax.set_title('Global Feature Importance (SHAP)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/shap_global_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Global: Beeswarm (how each feature affects risk) ─────
fig = plt.figure(figsize=(10, 7))
shap.plots.beeswarm(shap_values, show=False)
plt.title('SHAP Beeswarm — Feature Impact Direction', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Patient-level: Waterfall for 3 example patients ──────
for i, label in enumerate(['low_risk', 'high_risk', 'example']):
    # Pick a specific patient for demonstration
    idx = i * 25
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title(f'Patient {idx+1} — Individual Risk Explanation', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'outputs/shap_patient_{label}.png', dpi=150, bbox_inches='tight')
    plt.close()

# ── Save SHAP values to CSV for dashboard ─────────────────
shap_df = pd.DataFrame(shap_values.values, columns=FEATURES)
shap_df.to_csv('outputs/shap_values.csv', index=False)
print("   ✅ SHAP analysis complete")

# ═══════════════════════════════════════════════════════════
# PART 6 — FAIRLEARN BIAS AUDIT
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("RUNNING FAIRLEARN BIAS AUDIT...")

# Define bias metrics
metrics = {
    'accuracy':          accuracy_score,
    'selection_rate':    selection_rate,
    'false_pos_rate':    false_positive_rate,
    'false_neg_rate':    false_negative_rate,
}

# Run across all ethnicity subgroups
mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=eth_test
)

# AUC by subgroup (separate loop)
auc_by_group = {}
for eth in eth_test.unique():
    mask = eth_test == eth
    if mask.sum() >= 10 and y_test[mask].nunique() > 1:
        auc_by_group[eth] = roc_auc_score(y_test[mask], y_pred_proba[mask])

# ── Print bias report ──────────────────────────────────
print("\n   FAIRNESS METRICS BY ETHNICITY:")
print("   " + "─" * 50)
print(f"   {'Subgroup':<12} {'AUC':>6} {'Accuracy':>10} {'Sel.Rate':>10} {'FPR':>8} {'FNR':>8}")
print("   " + "─" * 50)
for eth in mf.by_group.index:
    row = mf.by_group.loc[eth]
    auc_val = auc_by_group.get(eth, float('nan'))
    print(f"   {eth:<12} {auc_val:>6.3f} {row['accuracy']:>10.3f} "
          f"{row['selection_rate']:>10.3f} {row['false_pos_rate']:>8.3f} "
          f"{row['false_neg_rate']:>8.3f}")

print("\n   OVERALL DISPARITIES:")
print(f"   Max AUC gap:            {max(auc_by_group.values()) - min(auc_by_group.values()):.3f}")
print(f"   Max FNR gap:            {mf.difference()['false_neg_rate']:.3f}")
print(f"   Max selection rate gap: {mf.difference()['selection_rate']:.3f}")

# ── Visualise AUC by ethnicity ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

# Plot 1: AUC by group
groups = list(auc_by_group.keys())
aucs   = list(auc_by_group.values())
bars = axes[0].bar(groups, aucs, color=colors[:len(groups)], edgecolor='white', linewidth=1.5)
axes[0].axhline(y=0.8, color='black', linestyle='--', alpha=0.5, label='Acceptable threshold (0.80)')
axes[0].set_ylim(0.5, 1.0)
axes[0].set_title('AUC by Ethnic Group', fontsize=13, fontweight='bold')
axes[0].set_ylabel('AUROC')
axes[0].legend()
for bar, val in zip(bars, aucs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: False Negative Rate (missed diagnoses) by group
fnr_vals = mf.by_group['false_neg_rate']
bars2 = axes[1].bar(fnr_vals.index, fnr_vals.values, color=colors[:len(fnr_vals)], edgecolor='white', linewidth=1.5)
axes[1].set_title('False Negative Rate by Ethnic Group\n(Higher = More Missed Diagnoses)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('False Negative Rate')
for bar, val in zip(bars2, fnr_vals.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Fairlearn Bias Audit — Preeclampsia Model', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/bias_audit.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Save bias table to CSV ────────────────────────────
bias_table = mf.by_group.copy()
bias_table['AUC'] = pd.Series(auc_by_group)
bias_table.to_csv('outputs/bias_report.csv')
print("\n   ✅ Bias audit complete")

# ═══════════════════════════════════════════════════════════
# PART 7 — SAVE MODEL
# ═══════════════════════════════════════════════════════════
joblib.dump(model,    'models/xgboost_preeclampsia.pkl')
joblib.dump(explainer,'models/shap_explainer.pkl')
joblib.dump(le,       'models/label_encoder.pkl')

# Save test predictions for dashboard
test_df = X_test.copy()
test_df['ethnicity']    = eth_test.values
test_df['y_true']       = y_test.values
test_df['y_pred_proba'] = y_pred_proba
test_df.to_csv('outputs/test_predictions.csv', index=False)

print("\n" + "=" * 55)
print("✅ ALL DONE — CHECK YOUR outputs/ FOLDER")
print("=" * 55)
print("\n   Files saved:")
print("   outputs/roc_curve.png")
print("   outputs/calibration_plot.png")
print("   outputs/shap_global_importance.png")
print("   outputs/shap_beeswarm.png")
print("   outputs/bias_audit.png")
print("   outputs/bias_report.csv")
print("   outputs/model_metrics.txt")
print("   models/xgboost_preeclampsia.pkl")
print("\n   ▶ Now run: streamlit run app.py")
