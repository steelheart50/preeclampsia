# ============================================================
# FILE 1: generate_data.py
# PURPOSE: Creates 1000 realistic fake patient records
#          based on real statistics from published papers
# HOW TO RUN: python generate_data.py
# ============================================================

import numpy as np
import pandas as pd
import os

# Set seed so results are the same every time you run
np.random.seed(42)

N = 1000  # number of patients to generate

print("Generating synthetic preeclampsia dataset...")

# ── ETHNICITY ──────────────────────────────────────────────
# Based on UK FMF screening study proportions
ethnicities = np.random.choice(
    ['White', 'Black', 'Asian', 'Hispanic'],
    size=N,
    p=[0.55, 0.20, 0.15, 0.10]  # real-world proportions from ASPRE trial
)

# ── AGE ────────────────────────────────────────────────────
# Mean and SD differ by ethnicity (from literature)
age = np.where(ethnicities == 'Black',
               np.random.normal(29, 6, N),   # Black women: younger on avg
       np.where(ethnicities == 'Asian',
               np.random.normal(28, 5, N),
       np.where(ethnicities == 'Hispanic',
               np.random.normal(27, 5, N),
               np.random.normal(31, 6, N)))) # White women
age = np.clip(age, 16, 50).astype(int)

# ── BMI ────────────────────────────────────────────────────
# Black women have higher mean BMI on average (documented in literature)
bmi = np.where(ethnicities == 'Black',
               np.random.normal(30.2, 6.5, N),
       np.where(ethnicities == 'Asian',
               np.random.normal(24.8, 4.5, N),
       np.where(ethnicities == 'Hispanic',
               np.random.normal(27.5, 5.5, N),
               np.random.normal(26.1, 5.8, N))))
bmi = np.clip(bmi, 16, 55).round(1)

# ── NULLIPAROUS (first pregnancy = 1) ──────────────────────
nulliparous = np.random.binomial(1, 0.48, N)

# ── CHRONIC HYPERTENSION ───────────────────────────────────
# Black women have higher rates (well-documented disparity)
chronic_htn = np.where(ethnicities == 'Black',
                        np.random.binomial(1, 0.15, N),   # 15% in Black women
                        np.random.binomial(1, 0.05, N))   # 5% in others

# ── DIABETES ───────────────────────────────────────────────
diabetes = np.where(ethnicities == 'Black',
                    np.random.binomial(1, 0.10, N),
           np.where(ethnicities == 'Asian',
                    np.random.binomial(1, 0.12, N),
                    np.random.binomial(1, 0.06, N)))

# ── KIDNEY DISEASE ─────────────────────────────────────────
kidney_disease = np.random.binomial(1, 0.03, N)

# ── MEAN ARTERIAL PRESSURE (MAP) ───────────────────────────
# mmHg — higher in hypertensive and Black patients
base_map = np.random.normal(82, 10, N)
map_val = base_map + chronic_htn * 12 + (ethnicities == 'Black') * 3
map_val = np.clip(map_val, 55, 130).round(1)

# ── UTERINE ARTERY PULSATILITY INDEX (UtAPI) ───────────────
# Key marker from FMF first-trimester screening
utapi = np.random.normal(1.58, 0.55, N)
utapi = np.clip(utapi, 0.4, 4.5).round(2)

# ── PlGF (Placental Growth Factor) ─────────────────────────
# pg/mL — lower values indicate higher risk
plgf_base = np.random.normal(45, 20, N)
plgf = plgf_base - chronic_htn * 8 - diabetes * 5
plgf = np.clip(plgf, 5, 150).round(1)

# ── sFlt-1 ─────────────────────────────────────────────────
# pg/mL — higher values indicate higher risk
sflt1 = np.random.normal(1200, 400, N)
sflt1 = np.clip(sflt1, 200, 4000).round(0)

# ── URIC ACID ──────────────────────────────────────────────
# mg/dL
uric_acid = np.random.normal(4.5, 1.2, N)
uric_acid = np.clip(uric_acid, 1.5, 10).round(1)

# ── OUTCOME: PREECLAMPSIA ──────────────────────────────────
# Real-world incidence ~5-8% overall, higher in Black women
# We build a logistic risk function from known predictors

log_odds = (
    -3.5                              # baseline
    + 0.04 * (age - 28)              # age effect
    + 0.05 * (bmi - 26)              # BMI effect
    + 0.8  * nulliparous             # first pregnancy
    + 1.4  * chronic_htn             # strong predictor
    + 0.9  * diabetes                # moderate predictor
    + 1.2  * kidney_disease          # strong predictor
    + 0.05 * (map_val - 82)          # MAP effect
    + 0.6  * (utapi - 1.58)          # UtAPI effect
    - 0.02 * (plgf - 45)             # protective effect
    + 0.001* (sflt1 - 1200)          # risk marker
    + 0.3  * (ethnicities == 'Black') # documented disparity
    + 0.15 * (ethnicities == 'Hispanic')
)

probability = 1 / (1 + np.exp(-log_odds))
preeclampsia = np.random.binomial(1, probability, N)

# ── ASSEMBLE INTO DATAFRAME ────────────────────────────────
df = pd.DataFrame({
    'patient_id':     range(1, N + 1),
    'age':            age,
    'bmi':            bmi,
    'ethnicity':      ethnicities,
    'nulliparous':    nulliparous,
    'chronic_htn':    chronic_htn,
    'diabetes':       diabetes,
    'kidney_disease': kidney_disease,
    'map':            map_val,
    'utapi':          utapi,
    'plgf':           plgf,
    'sflt1':          sflt1,
    'uric_acid':      uric_acid,
    'preeclampsia':   preeclampsia
})

# ── SAVE ───────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
df.to_excel('data/preeclampsia_dataset.xlsx', index=False)
df.to_csv('data/preeclampsia_dataset.csv', index=False)

# ── SUMMARY REPORT ─────────────────────────────────────────
print("\n✅ Dataset created successfully!")
print(f"   Total patients:       {len(df)}")
print(f"   Preeclampsia cases:   {df['preeclampsia'].sum()} ({df['preeclampsia'].mean()*100:.1f}%)")
print(f"\n   Cases by ethnicity:")
for eth in ['White', 'Black', 'Asian', 'Hispanic']:
    sub = df[df['ethnicity'] == eth]
    rate = sub['preeclampsia'].mean() * 100
    print(f"   {eth:10s}:  {rate:.1f}%  (n={len(sub)})")
print(f"\n   Saved to: data/preeclampsia_dataset.xlsx")
print(f"             data/preeclampsia_dataset.csv")
