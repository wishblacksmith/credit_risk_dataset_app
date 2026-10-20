import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_models():
    model = joblib.load('credit_model.pkl')
    scaler = joblib.load('credit_scaler.pkl')
    return model, scaler

model, scaler = load_models()

st.title("🏦 Brezilya Kredi Riski ve Temerrüt Tahmini (Credit Risk AI)")
st.write("Bu uygulama, XGBoost ve Makine Öğrenmesi kullanarak bireysel kredi başvurularının risk durumunu (temerrüt ihtimalini) analiz eder.")

st.subheader("Kredi Başvuru Bilgilerini Girin:")

person_age = st.number_input("Başvuru Sahibi Yaşı (Age)", min_value=18, max_value=100, value=30)
person_income = st.number_input("Yıllık Gelir (Income - $)", min_value=0.0, max_value=2000000.0, value=50000.0)
person_emp_length = st.number_input("Çalışma Süresi (Yıl)", min_value=0.0, max_value=50.0, value=5.0)
loan_amnt = st.number_input("Talep Edilen Kredi Miktarı (Loan Amount - $)", min_value=100.0, max_value=100000.0, value=10000.0)
loan_int_rate = st.number_input("Kredi Faiz Oranı (%)", min_value=0.0, max_value=30.0, value=11.0)
loan_percent_income = st.number_input("Gelire Oranı (Loan Percent Income)", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Kredi Geçmişi Süresi (Yıl)", min_value=0.0, max_value=30.0, value=4.0)

home_ownership = st.selectbox("Ev Sahipliği Durumu (Home Ownership)", ["RENT", "MORTGAGE", "OWN", "OTHER"])
loan_intent = st.selectbox("Kredi Kullanım Amacı (Intent)", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Kredi Notu Derecesi (Loan Grade)", ["A", "B", "C", "D", "E", "F", "G"])
default_on_file = st.selectbox("Geçmişte Temerrüt Kaydı Var mı? (Default on File)", ["N", "Y"])

if st.button("Kredi Riskini Analiz Et"):
    input_dict = {
        'person_age': [person_age],
        'person_income': [person_income],
        'person_emp_length': [person_emp_length],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'person_home_ownership_OTHER': [1 if home_ownership == 'OTHER' else 0],
        'person_home_ownership_OWN': [1 if home_ownership == 'OWN' else 0],
        'person_home_ownership_RENT': [1 if home_ownership == 'RENT' else 0],
        'loan_intent_EDUCATION': [1 if loan_intent == 'EDUCATION' else 0],
        'loan_intent_HOMEIMPROVEMENT': [1 if loan_intent == 'HOMEIMPROVEMENT' else 0],
        'loan_intent_MEDICAL': [1 if loan_intent == 'MEDICAL' else 0],
        'loan_intent_PERSONAL': [1 if loan_intent == 'PERSONAL' else 0],
        'loan_intent_VENTURE': [1 if loan_intent == 'VENTURE' else 0],
        'loan_grade_B': [1 if loan_grade == 'B' else 0],
        'loan_grade_C': [1 if loan_grade == 'C' else 0],
        'loan_grade_D': [1 if loan_grade == 'D' else 0],
        'loan_grade_E': [1 if loan_grade == 'E' else 0],
        'loan_grade_F': [1 if loan_grade == 'F' else 0],
        'loan_grade_G': [1 if loan_grade == 'G' else 0],
        'cb_person_default_on_file_Y': [1 if default_on_file == 'Y' else 0]
    }
    
    input_df = pd.DataFrame(input_dict)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    risk_prob = probability[0][1] * 100
    safe_prob = probability[0][0] * 100
    
    if prediction[0] == 1:
        st.error(f"🚨 YÜKSEK RİSK / TEMERRÜT İHTİMALİ! Bu kredinin geri ödenmeme riski yüksektir. (Risk Oranı: %{risk_prob:.2f})")
    else:
        st.success(f"✅ KREDİ ONAYLANABİLİR (GÜVENLİ). Başvuru sahibinin temerrüt riski düşüktür. (Güven Oranı: %{safe_prob:.2f})")
