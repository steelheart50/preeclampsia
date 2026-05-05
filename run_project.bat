@echo off
echo ================================================
echo  PREECLAMPSIA PROJECT — FULL PIPELINE RUNNER
echo ================================================
echo.

echo [STEP 1] Activating virtual environment...
call venv\Scripts\activate
echo Done.
echo.

echo [STEP 2] Installing all required libraries...
pip install pandas numpy scikit-learn xgboost shap fairlearn streamlit openpyxl matplotlib seaborn imbalanced-learn joblib -q
echo Done.
echo.

echo [STEP 3] Generating synthetic patient data...
python generate_data.py
echo.

echo [STEP 4] Training model + SHAP + Fairlearn analysis...
python train_model.py
echo.

echo [STEP 5] Launching Streamlit dashboard...
echo  Open your browser at: http://localhost:8501
streamlit run app.py
