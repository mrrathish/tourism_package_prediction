import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="Retheesh/tourism-customer-prediction-model", filename="tourism_customer_prediction_model.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer will purchase the Wellness Tourism Package
based on their characteristics and interaction data.
Please enter the customer details below to get a prediction.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    preferred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=1)

with col2:
    st.subheader("Additional Information")
    passport = st.selectbox("Has Passport", [0, 1])
    own_car = st.selectbox("Owns Car", [0, 1])
    num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (USD)", min_value=0, max_value=50000, value=15000)

    st.subheader("Interaction Details")
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)

# Mapping for categorical variables
contact_mapping = {"Company Invited": 0, "Self Inquiry": 1}
occupation_mapping = {"Salaried": 0, "Small Business": 1, "Large Business": 2, "Free Lancer": 3}
gender_mapping = {"Male": 0, "Female": 1}
marital_mapping = {"Single": 0, "Married": 1, "Divorced": 2}
designation_mapping = {"Executive": 0, "Manager": 1, "Senior Manager": 2, "AVP": 3, "VP": 4}
product_mapping = {"Basic": 0, "Deluxe": 1, "Standard": 2, "Super Deluxe": 3, "King": 4}

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': contact_mapping[typeof_contact],
    'CityTier': city_tier,
    'Occupation': occupation_mapping[occupation],
    'Gender': gender_mapping[gender],
    'NumberOfPersonVisiting': num_persons_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_mapping[marital_status],
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation_mapping[designation],
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction,
    'ProductPitched': product_mapping[product_pitched],
    'NumberOfFollowups': num_followups,
    'DurationOfPitch': duration_pitch
}])

# Predict button
if st.button("Predict Purchase Probability"):
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success(f"Great!! This customer is LIKELY to purchase the Wellness Tourism Package")
        st.info(f"Purchase Probability: {prediction_proba:.2%}")
    else:
        st.warning(f"Am skeptic that this customer will purchase the Wellness Tourism Package")
        st.info(f"Purchase Probability: {prediction_proba:.2%}")

    # Show confidence level
    confidence = "High" if prediction_proba > 0.7 else "Medium" if prediction_proba > 0.5 else "Low"
    st.write(f"**Confidence Level:** {confidence}")
