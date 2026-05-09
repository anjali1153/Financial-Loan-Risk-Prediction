import streamlit as st
import joblib
import pandas as pd

# Load model
pipeline = joblib.load('model.pkl')

# Page config
st.set_page_config(page_title="Loan Risk Prediction", page_icon="💰")

# Title
st.title("Financial Loan Risk Prediction 💰")

# Optional GIF
st.image("loan.gif", width='stretch')


# ================= INPUT FIELDS =================

loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Education", "Business"])

term = st.selectbox("Loan Term", ["Short", "Medium", "Long"])

business_or_commercial = st.selectbox("Business or Commercial", ["Yes", "No"])

loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"])

loan_amount = st.number_input("Loan Amount", 0.0, 10000000.0)

co_applicant_credit_type = st.selectbox("Co-applicant Credit Type", ["Good", "Bad", "Unknown"])

income = st.number_input("Income", 0.0, 10000000.0)

property_value = st.number_input("Property Value", 0.0, 10000000.0)

LTV = st.number_input("Loan to Value Ratio (LTV)", 0.0, 100.0)

dtir1 = st.number_input("Debt-to-Income Ratio", 0.0, 100.0)

credit_type = st.selectbox("Credit Type", ["Good", "Bad", "Unknown"])

Upfront_charges = st.number_input("Upfront Charges", 0.0, 1000000.0)

rate_of_interest = st.number_input("Rate of Interest", 0.0, 20.0)

Interest_rate_spread = st.number_input("Interest Rate Spread", 0.0, 10.0)


# ================= DATAFRAME =================

input_data = pd.DataFrame([[loan_purpose, term, business_or_commercial, loan_type,
                            loan_amount, co_applicant_credit_type, income,
                            property_value, LTV, dtir1, credit_type,
                            Upfront_charges, rate_of_interest, Interest_rate_spread]],
                          columns=['loan_purpose', 'term', 'business_or_commercial',
                                   'loan_type', 'loan_amount',
                                   'co-applicant_credit_type', 'income',
                                   'property_value', 'LTV', 'dtir1',
                                   'credit_type', 'Upfront_charges',
                                   'rate_of_interest', 'Interest_rate_spread'])


# ================= PREDICTION =================

if st.button("Predict Risk"):
    pred = pipeline.predict(input_data)

    if pred[0] == 1:
        st.metric(label="Loan Risk", value="⚠️ High Risk")
    else:
        st.metric(label="Loan Risk", value="✅ Low Risk")


# Footer
st.markdown("---")