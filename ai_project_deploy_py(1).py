import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_random_forest.pkl")

model = load_model()

st.title("Customer Churn Prediction App")
st.write("Fill customer details to predict churn.")

# IMPORTANT:
# These must match your training dataset columns exactly

gender = st.selectbox("Gender", ["Male", "Female"])
status = st.selectbox("Status", ["Single", "Married"])
children = st.number_input("Children", min_value=0, max_value=10, value=0)
est_income = st.number_input("Est Income", min_value=0.0, value=50000.0)
car_owner = st.selectbox("Car Owner", ["Yes", "No"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
paymethod = st.selectbox("Paymethod", ["Credit Card", "Debit Card", "Cash", "Online"])
usage = st.number_input("Usage", min_value=0.0, value=100.0)
rateplan = st.selectbox("RatePlan", ["Basic", "Premium", "Unlimited"])

if st.button("Predict"):

    input_data = {
        "Gender": gender,
        "Status": status,
        "Children": children,
        "Est Income": est_income,
        "Car Owner": car_owner,
        "Age": age,
        "Paymethod": paymethod,
        "Usage": usage,
        "RatePlan": rateplan
    }

    input_df = pd.DataFrame([input_data])

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
