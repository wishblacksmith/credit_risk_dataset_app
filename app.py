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
st.write("Bu uygulama, XGBoost kullanarak bireysel kredi başvurularının risk durumunu anında analiz eder.")

# Session State (Hafıza) Başlatma
if 'd_age' not in st.session_state:
    st.session_state.d_age = 30
    st.session_state.d_income = 50000.0
    st.session_state.d_emp = 5.0
    st.session_state.d_amount = 10000.0
    st.session_state.d_rate = 11.0
    st.session_state.d_pct = 0.2
    st.session_state.d_hist = 4.0
    st.session_state.d_home = "RENT"
    st.session_state.d_intent = "PERSONAL"
    st.session_state.d_grade = "A"
    st.session_state.d_default = "N"

st.subheader("Hızlı Test Senaryoları:")
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🟢 Normal Başvuru Örneği Yükle"):
        st.session_state.d_age = 35
        st.session_state.d_income = 75000.0
        st.session_state.d_emp = 8.0
        st.session_state.d_amount = 8000.0
        st.session_state.d_rate = 8.5
        st.session_state.d_pct = 0.1
        st.session_state.d_hist = 8.0
        st.session_state.d_home = "MORTGAGE"
        st.session_state.d_intent = "VENTURE"
        st.session_state.d_grade = "A"
        st.session_state.d_default = "N"

with col_btn2:
    if st.button("🔴 Riskli / Temerrüt Örneği Yükle"):
        st.session_state.d_age = 22
        st.session_state.d_income = 24000.0
        st.session_state.d_emp = 1.0
        st.session_state.d_amount = 18000.0
        st.session_state.d_rate = 20.2
        st.session_state.d_pct = 0.75
        st.session_state.d_hist = 2.0
        st.session_state.d_home = "RENT"
        st.session_state.d_intent = "MEDICAL"
        st.session_state.d_grade = "D"
        st.session_state.d_default = "Y"

st.subheader("Kredi Başvuru Bilgilerini Düzenle:")

person_age = st.number_input("Başvuru Sahibi Yaşı", min_value=18, max_value=100, value=st.session_state.d_age)
person_income = st.number_input("Yıllık Gelir ($)", min_value=0.0, max_value=2000000.0, value=st.session_state.d_income)
person_emp_length = st.number_input("Çalışma Süresi (Yıl)", min_value=0.0, max_value=50.0, value=st.session_state.d_emp)
loan_amnt = st.number_input("Talep Edilen Kredi Miktarı ($)", min_value=100.0, max_value=100000.0, value=st.session_state.d_amount)
loan_int_rate = st.number_input("Kredi Faiz Oranı (%)", min_value=0.0, max_value=30.0, value=st.session_state.d_rate)
loan_percent_income = st.number_input("Gelire Oranı", min_value=0.0, max_value=1.0, value=st.session_state.d_pct)
cb_person_cred_hist_length = st.number_input("Kredi Geçmişi Süresi (Yıl)", min_value=0.0, max_value=30.0, value=st.session_state.d_hist)

home_ownership = st.selectbox("Ev Sahipliği Durumu", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(st.session_state.d_home))
loan_intent = st.selectbox("Kredi Kullanım Amacı", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], index=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"].index(st.session_state.d_intent))
loan_grade = st.selectbox("Kredi Notu Derecesi", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(st.session_state.d_grade))
default_on_file = st.selectbox("Geçmişte Temerrüt Kaydı Var mı?", ["N", "Y"], index=["N", "Y"].index(st.session_state.d_default))

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
    
    probability = model.predict_proba(input_scaled)
    risk_prob = probability[0][1] * 100
    safe_prob = probability[0][0] * 100
    
    if risk_prob > 20.0:
        st.error(f"🚨 YÜKSEK RİSK / TEMERRÜT İHTİMALİ! Bu kredinin geri ödenmeme riski yüksektir. (Risk Oranı: %{risk_prob:.2f})")
    else:
        st.success(f"✅ KREDİ ONAYLANABİLİR (GÜVENLİ). Başvuru sahibinin temerrüt riski düşüktür. (Güven Oranı: %{safe_prob:.2f})")
