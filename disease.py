import streamlit as st
import numpy as np
import joblib

# Load all models and scalers
parkinsons_model = joblib.load("parkinsons_rf_model (1).pkl")
parkinsons_scaler = joblib.load("scaler (2).pkl")

liver_model = joblib.load("liver_rf_model.pkl")
liver_scaler = joblib.load("scaler_liver.pkl")

kidney_model = joblib.load("kidney_rf_model.pkl")
kidney_scaler = joblib.load("kidney_scaler.pkl")

# Set page configuration
st.set_page_config(page_title="Multiple Disease Predictor", layout="centered")
st.title("🩺 Multiple Disease Prediction System")
st.markdown("Predict **Parkinson's**, **Liver**, or **Kidney Disease** using pre-trained ML models.")

# Sidebar for selection
disease = st.sidebar.selectbox("Select Disease to Predict", ["Parkinson's", "Liver", "Kidney"])

# ===================== PARKINSON'S =====================
if disease == "Parkinson's":
    st.header("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter the following biomedical voice measurements:")

    # Input fields for Parkinson's
    parkinsons_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
        "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    inputs = [st.number_input(label, format="%.6f") for label in parkinsons_features]

    if st.button("Predict Parkinson's"):
        scaled = parkinsons_scaler.transform([inputs])
        result = parkinsons_model.predict(scaled)
        st.success("🧠 Parkinson's Detected!" if result[0] == 1 else "✅ Healthy")

# ===================== LIVER =====================
elif disease == "Liver":
    st.header("🩸 Liver Disease Prediction")
    st.markdown("Provide your clinical liver panel values:")

    # Input fields for Liver
    age = st.number_input("Age", min_value=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin")
    direct_bilirubin = st.number_input("Direct Bilirubin")
    alk_phos = st.number_input("Alkaline Phosphotase")
    sgot = st.number_input("Alamine Aminotransferase (SGPT)")
    sgpt = st.number_input("Aspartate Aminotransferase (SGOT)")
    total_proteins = st.number_input("Total Proteins")
    albumin = st.number_input("Albumin")
    ag_ratio = st.number_input("Albumin and Globulin Ratio")

    gender_encoded = 1 if gender == "Male" else 0
    liver_input = [age, gender_encoded, total_bilirubin, direct_bilirubin,
                   alk_phos, sgot, sgpt, total_proteins, albumin, ag_ratio]

    if st.button("Predict Liver Disease"):
        scaled = liver_scaler.transform([liver_input])
        result = liver_model.predict(scaled)
        st.success("🩸 Liver Disease Detected!" if result[0] == 1 else "✅ Healthy")

# ===================== KIDNEY =====================
elif disease == "Kidney":
    st.header("🧪 Kidney Disease Prediction")
    st.markdown("Enter your lab values:")

    # These should match your kidney dataset's feature order
    age = st.number_input("Age")
    bp = st.number_input("Blood Pressure")
    sg = st.number_input("Specific Gravity")
    al = st.number_input("Albumin")
    su = st.number_input("Sugar")
    rbc = st.selectbox("RBC", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    bgr = st.number_input("Blood Glucose Random")
    bu = st.number_input("Blood Urea")
    sc = st.number_input("Serum Creatinine")
    sod = st.number_input("Sodium")
    pot = st.number_input("Potassium")
    hemo = st.number_input("Hemoglobin")
    htn=st.selectbox("hypertension", ["yes", "no"])
    # Example encoded inputs (simplified)
    rbc_encoded = 1 if rbc == "normal" else 0
    pc_encoded = 1 if pc == "normal" else 0
    htn_encoded = 1 if htn == "yes" else 0

    kidney_input = [age, bp, sg, al, su, rbc_encoded, pc_encoded,
                bgr, bu, sc, sod, pot, hemo,htn_encoded]

    if st.button("Predict Kidney Disease"):
        scaled = kidney_scaler.transform([kidney_input])
        result = kidney_model.predict(scaled)
        st.success("🧪 Kidney Disease Detected!" if result[0] == 1 else "✅ Healthy")