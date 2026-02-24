import streamlit as st
import joblib
import pandas as pd

# -----------------------
# Load trained pipeline
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("customer_churn_random_forest.pkl")

model = load_model()

st.title("Customer Churn Prediction App")
st.write("Fill customer details to predict churn.")

# Show expected columns (for debugging – can remove later)
st.write("Model expects these features:")
st.write(model.feature_names_in_)

# -----------------------
# User Inputs
# -----------------------

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=600.0)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "InternetService",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.selectbox(
    "PaymentMethod",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

online_security = st.selectbox("OnlineSecurity", ["Yes", "No"])
tech_support = st.selectbox("TechSupport", ["Yes", "No"])
paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])

# -----------------------
# Prediction
# -----------------------

if st.button("Predict"):

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "PaperlessBilling": paperless_billing
    }

    input_df = pd.DataFrame([input_data])

    # FORCE column alignment with training schema
    input_df = input_df.reindex(columns=model.feature_names_in_)

    prediction = model.predict(input_df)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is likely to STAY")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)
        confidence = max(probability[0]) * 100
        st.write(f"Confidence: {confidence:.2f}%")
