# ============================================================
# FILE 3: app.py
# PURPOSE: Interactive Streamlit dashboard — clinicians
#          enter a patient and see risk + bias analysis
# HOW TO RUN: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Preeclampsia Risk Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #C62828;
        margin-bottom: 0; letter-spacing: -0.5px;
    }
    .subtitle { color: #666; font-size: 1rem; margin-top: 0; }
    .risk-box-high {
        background: #FFEBEE; border-left: 6px solid #C62828;
        padding: 20px; border-radius: 8px; margin: 10px 0;
    }
    .risk-box-low {
        background: #E8F5E9; border-left: 6px solid #2E7D32;
        padding: 20px; border-radius: 8px; margin: 10px 0;
    }
    .metric-box {
        background: #F5F5F5; padding: 12px 18px;
        border-radius: 8px; text-align: center;
    }
    .bias-warning {
        background: #FFF3E0; border-left: 6px solid #E65100;
        padding: 16px; border-radius: 8px; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model and data ───────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('models/xgboost_preeclampsia.pkl')
    explainer= joblib.load('models/shap_explainer.pkl')
    le       = joblib.load('models/label_encoder.pkl')
    return model, explainer, le

@st.cache_data
def load_results():
    preds = pd.read_csv('outputs/test_predictions.csv')
    bias  = pd.read_csv('outputs/bias_report.csv', index_col=0)
    return preds, bias

FEATURES = ['age', 'bmi', 'ethnicity_encoded', 'nulliparous',
            'chronic_htn', 'diabetes', 'kidney_disease',
            'map', 'utapi', 'plgf', 'sflt1', 'uric_acid']

FEATURE_LABELS = {
    'age': 'Age (years)',
    'bmi': 'BMI',
    'ethnicity_encoded': 'Ethnicity',
    'nulliparous': 'First Pregnancy',
    'chronic_htn': 'Chronic Hypertension',
    'diabetes': 'Diabetes',
    'kidney_disease': 'Kidney Disease',
    'map': 'Mean Arterial Pressure',
    'utapi': 'Uterine Artery PI',
    'plgf': 'PlGF (pg/mL)',
    'sflt1': 'sFlt-1 (pg/mL)',
    'uric_acid': 'Uric Acid (mg/dL)'
}

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🩺 Preeclampsia Risk Prediction Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered risk scoring with explainability and bias audit | Research Prototype</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Check model exists ────────────────────────────────────
if not os.path.exists('models/xgboost_preeclampsia.pkl'):
    st.error("⚠️ Model not found. Please run: `python train_model.py` first.")
    st.stop()

model, explainer, le = load_model()
preds_df, bias_df = load_results()

# ═══════════════════════════════════════════════════════════
# SIDEBAR — PATIENT INPUT
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🧑‍⚕️ Enter Patient Data")
    st.markdown("Adjust sliders for the patient you want to assess.")
    st.markdown("---")

    age        = st.slider("Age (years)", 16, 50, 28)
    bmi        = st.slider("BMI", 16.0, 55.0, 26.0, step=0.1)
    ethnicity  = st.selectbox("Ethnicity", ['White', 'Black', 'Asian', 'Hispanic'])
    nulliparous= st.selectbox("First pregnancy?", ['Yes', 'No'])
    chronic_htn= st.selectbox("Chronic hypertension?", ['No', 'Yes'])
    diabetes   = st.selectbox("Diabetes?", ['No', 'Yes'])
    kidney_dis = st.selectbox("Kidney disease?", ['No', 'Yes'])

    st.markdown("---")
    st.subheader("📊 Clinical Measurements")
    map_val    = st.slider("Mean Arterial Pressure (mmHg)", 55, 130, 82)
    utapi      = st.slider("Uterine Artery PI", 0.4, 4.5, 1.5, step=0.01)
    plgf       = st.slider("PlGF (pg/mL)", 5.0, 150.0, 45.0, step=0.5)
    sflt1      = st.slider("sFlt-1 (pg/mL)", 200, 4000, 1200, step=10)
    uric_acid  = st.slider("Uric Acid (mg/dL)", 1.5, 10.0, 4.5, step=0.1)

    predict_btn = st.button("🔍 Calculate Risk", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Assessment",
    "🔬 Explainability (SHAP)",
    "⚖️ Bias Audit",
    "📈 Model Performance"
])

# ── Build patient feature vector ──────────────────────────
eth_encoded   = le.transform([ethnicity])[0]
null_bin      = 1 if nulliparous == 'Yes' else 0
htn_bin       = 1 if chronic_htn == 'Yes' else 0
diab_bin      = 1 if diabetes == 'Yes' else 0
kidney_bin    = 1 if kidney_dis == 'Yes' else 0

patient = pd.DataFrame([[
    age, bmi, eth_encoded, null_bin, htn_bin, diab_bin,
    kidney_bin, map_val, utapi, plgf, sflt1, uric_acid
]], columns=FEATURES)

risk_score = model.predict_proba(patient)[0][1]

# ── Uncertainty interval (bootstrap approximation) ────────
np.random.seed(42)
bootstrap_preds = []
for _ in range(200):
    noise = np.random.normal(0, 0.02, patient.shape)
    perturbed = patient + noise
    perturbed = perturbed.clip(lower=0)
    bootstrap_preds.append(model.predict_proba(perturbed)[0][1])
ci_low  = np.percentile(bootstrap_preds, 5)
ci_high = np.percentile(bootstrap_preds, 95)

# ═══════════════════════════════════════════════════════════
# TAB 1 — RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3 = st.columns([1, 1, 1])

    risk_pct = risk_score * 100
    risk_label = "HIGH RISK" if risk_score >= 0.5 else ("MODERATE RISK" if risk_score >= 0.2 else "LOW RISK")
    box_class  = "risk-box-high" if risk_score >= 0.2 else "risk-box-low"
    color      = "#C62828" if risk_score >= 0.5 else ("#E65100" if risk_score >= 0.2 else "#2E7D32")

    with col1:
        st.markdown(f"""
        <div class="{box_class}">
            <h2 style="color:{color}; margin:0">{risk_label}</h2>
            <h1 style="color:{color}; font-size:3rem; margin:5px 0">{risk_pct:.1f}%</h1>
            <p style="margin:0; color:#555">Predicted preeclampsia risk</p>
            <p style="margin:4px 0; font-size:0.9rem; color:#777">
                90% CI: {ci_low*100:.1f}% – {ci_high*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Patient Summary**")
        st.markdown(f"""
        | Feature | Value |
        |---------|-------|
        | Age | {age} years |
        | BMI | {bmi} |
        | Ethnicity | {ethnicity} |
        | First pregnancy | {nulliparous} |
        | Hypertension | {chronic_htn} |
        | Diabetes | {diabetes} |
        | MAP | {map_val} mmHg |
        | UtAPI | {utapi} |
        | PlGF | {plgf} pg/mL |
        """)

    with col3:
        st.markdown("**Risk Gauge**")
        fig, ax = plt.subplots(figsize=(4, 2.5))
        gauge_colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336']
        ax.barh(['Risk'], [0.2],  color=gauge_colors[0], height=0.4)
        ax.barh(['Risk'], [0.15], left=0.2, color=gauge_colors[1], height=0.4)
        ax.barh(['Risk'], [0.15], left=0.35, color=gauge_colors[2], height=0.4)
        ax.barh(['Risk'], [0.50], left=0.50, color=gauge_colors[3], height=0.4)
        ax.axvline(x=risk_score, color='black', linewidth=3, label=f'Patient: {risk_pct:.1f}%')
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.35, 0.5, 1.0])
        ax.set_xticklabels(['0%', '20%', '35%', '50%', '100%'], fontsize=8)
        ax.set_title('Risk Level', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Clinical interpretation ────────────────────────
    st.markdown("---")
    st.subheader("📋 Clinical Interpretation")

    if risk_score >= 0.5:
        st.error("""**High Risk** — This patient has substantial predicted risk of developing preeclampsia.
        Consider: increased monitoring, low-dose aspirin (if <16 weeks), early referral to maternal-fetal medicine.
        This is a research prototype and does not replace clinical judgement.""")
    elif risk_score >= 0.2:
        st.warning("""**Moderate Risk** — Elevated risk detected. Review individual predictors (see SHAP tab).
        Consider: baseline BP monitoring, repeat assessment at 20 weeks.
        This is a research prototype and does not replace clinical judgement.""")
    else:
        st.success("""**Low Risk** — Predicted risk is within normal range.
        Routine antenatal care recommended. Reassess if clinical status changes.
        This is a research prototype and does not replace clinical judgement.""")

# ═══════════════════════════════════════════════════════════
# TAB 2 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔬 Why did the model give this prediction?")
    st.markdown("Each bar below shows how much a predictor **increased** (red) or **decreased** (blue) the risk score.")

    shap_vals = explainer(patient)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Waterfall plot
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_vals[0], show=False,
                             max_display=12)
        plt.title(f"Patient Risk Explanation — {risk_pct:.1f}% Predicted Risk",
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**How to read this chart:**")
        st.markdown("""
        - **E[f(x)]** = average baseline risk of all patients
        - **f(x)** = this patient's predicted risk
        - 🔴 Red bars = this feature **increases** risk
        - 🔵 Blue bars = this feature **decreases** risk
        - Longer bar = bigger impact on the prediction

        ---
        **Top contributors for this patient:**
        """)

        shap_series = pd.Series(
            shap_vals.values[0],
            index=[FEATURE_LABELS.get(f, f) for f in FEATURES]
        ).sort_values(key=abs, ascending=False)

        for fname, fval in shap_series.head(5).items():
            direction = "↑ increases risk" if fval > 0 else "↓ decreases risk"
            arrow = "🔴" if fval > 0 else "🔵"
            st.markdown(f"{arrow} **{fname}**: {direction} ({fval:+.3f})")

    # Global importance
    st.markdown("---")
    st.subheader("📊 Global: Which features matter most overall?")
    if os.path.exists('outputs/shap_global_importance.png'):
        st.image('outputs/shap_global_importance.png', use_column_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — BIAS AUDIT
# ═══════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚖️ Fairness Analysis Across Ethnic Groups")
    st.markdown("""
    This section answers: **does the model perform equally well for all ethnic groups?**
    A fair model should have similar AUC and false negative rates across groups.
    Higher false negative rate = more missed diagnoses in that group.
    """)

    # Bias metrics table
    st.markdown("**Bias Metrics by Ethnic Group (Test Set)**")
    if 'AUC' in bias_df.columns:
        display_cols = [c for c in ['AUC', 'accuracy', 'selection_rate', 'false_pos_rate', 'false_neg_rate'] if c in bias_df.columns]
        st.dataframe(
            bias_df[display_cols].style
            .format({c: '{:.3f}' for c in display_cols})
            .background_gradient(subset=['false_neg_rate'], cmap='RdYlGn_r')
            .background_gradient(subset=['AUC'], cmap='RdYlGn'),
            use_container_width=True
        )

    # Bias chart
    if os.path.exists('outputs/bias_audit.png'):
        st.image('outputs/bias_audit.png', use_column_width=True)

    # ── Interpretation ─────────────────────────────────
    st.markdown("---")
    st.subheader("💡 What Does This Mean?")
    st.markdown("""
    <div class="bias-warning">
    <b>⚠️ Documented Disparity:</b> Black women are at higher absolute risk of preeclampsia,
    yet are more likely to be diagnosed later. If this model shows a lower AUC or higher false
    negative rate for Black patients, it reflects bias in the underlying data — not just the model.
    This is why bias auditing is essential before any clinical deployment.
    <br><br>
    <b>What to do about it:</b> Techniques like re-weighting training data,
    post-hoc threshold adjustment per group (equalized odds), or collecting
    more representative data can reduce disparity.
    </div>
    """, unsafe_allow_html=True)

    if ethnicity == 'Black' and risk_score < 0.3:
        st.warning("""⚠️ **Note:** This patient is Black. The model may underestimate risk for this
        subgroup. Consider applying a lower risk threshold for clinical intervention in Black patients.""")

# ═══════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Overall Model Performance")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('outputs/roc_curve.png'):
            st.image('outputs/roc_curve.png', caption='ROC Curve', use_column_width=True)
    with col2:
        if os.path.exists('outputs/calibration_plot.png'):
            st.image('outputs/calibration_plot.png', caption='Calibration Plot', use_column_width=True)

    if os.path.exists('outputs/model_metrics.txt'):
        st.markdown("**Performance Metrics**")
        with open('outputs/model_metrics.txt') as f:
            st.code(f.read())

    # Risk distribution by ethnicity
    st.markdown("---")
    st.subheader("Risk Score Distribution by Ethnic Group")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'White': '#2196F3', 'Black': '#F44336', 'Asian': '#4CAF50', 'Hispanic': '#FF9800'}
    for eth, grp in preds_df.groupby('ethnicity'):
        ax.hist(grp['y_pred_proba'], bins=25, alpha=0.6,
                label=eth, color=colors.get(eth, 'grey'), density=True)
    ax.set_xlabel('Predicted Risk Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Risk Score Distribution by Ethnicity', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#999; font-size:0.8rem'>
⚠️ Research prototype only. Not validated for clinical use.
Built with XGBoost · SHAP · Fairlearn · Streamlit
</p>
""", unsafe_allow_html=True)
