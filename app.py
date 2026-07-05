import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Load the model
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model('catboost_model.cbm')
    return model

model = load_model()

st.title("📱 Teen Phone Addiction Predictor")
st.markdown("Enter the teenager's daily habits and demographics below to predict their phone addiction level.")

# Group inputs into columns/expander for better UI
with st.sidebar:
    st.header("User Profile & Demographics")
    age = st.slider("Age", 10, 19, 15)
    gender = st.selectbox("Gender (Encoded)", [0, 1, 2], help="0=Female, 1=Male, 2=Other/Unspecified depending on encoding")
    school_grade = st.slider("School Grade (Encoded)", 0, 12, 9)
    
    st.header("Mental Health & Wellbeing")
    sleep_hours = st.slider("Sleep Hours", 2.0, 12.0, 7.0, 0.5)
    anxiety_level = st.slider("Anxiety Level", 0, 10, 5)
    depression_level = st.slider("Depression Level", 0, 10, 5)
    self_esteem = st.slider("Self Esteem", 0, 10, 5)

st.header("Digital Usage Habits")
col1, col2 = st.columns(2)

with col1:
    daily_usage_hours = st.slider("Daily Usage Hours", 0.0, 16.0, 5.0, 0.5)
    apps_used_daily = st.slider("Apps Used Daily", 1, 30, 10)
    phone_checks = st.slider("Phone Checks Per Day", 0, 200, 50)
    screen_time_bed = st.slider("Screen Time Before Bed (Hours)", 0.0, 5.0, 1.0, 0.5)
    weekend_usage = st.slider("Weekend Usage Hours", 0.0, 16.0, 6.0, 0.5)

with col2:
    time_social = st.slider("Time on Social Media (Hours)", 0.0, 10.0, 2.0, 0.5)
    time_gaming = st.slider("Time on Gaming (Hours)", 0.0, 10.0, 1.0, 0.5)
    time_education = st.slider("Time on Education (Hours)", 0.0, 10.0, 1.0, 0.5)
    phone_purpose = st.slider("Phone Usage Purpose (Encoded)", 0, 5, 1)

st.header("Lifestyle & Environment")
col3, col4 = st.columns(2)

with col3:
    academic_perf = st.slider("Academic Performance (0-100)", 0, 100, 75)
    exercise_hours = st.slider("Exercise Hours (Daily)", 0.0, 5.0, 1.0, 0.5)
    
with col4:
    social_interactions = st.slider("Social Interactions (0-10)", 0, 10, 5)
    parental_control = st.slider("Parental Control Level (0-10)", 0, 10, 5)
    family_comm = st.slider("Family Communication (0-10)", 0, 10, 5)

# Create input dataframe based on the exact training feature order
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'School_Grade': school_grade,
    'Daily_Usage_Hours': daily_usage_hours,
    'Sleep_Hours': sleep_hours,
    'Academic_Performance': academic_perf,
    'Social_Interactions': social_interactions,
    'Exercise_Hours': exercise_hours,
    'Anxiety_Level': anxiety_level,
    'Depression_Level': depression_level,
    'Self_Esteem': self_esteem,
    'Parental_Control': parental_control,
    'Screen_Time_Before_Bed': screen_time_bed,
    'Phone_Checks_Per_Day': phone_checks,
    'Apps_Used_Daily': apps_used_daily,
    'Time_on_Social_Media': time_social,
    'Time_on_Gaming': time_gaming,
    'Time_on_Education': time_education,
    'Phone_Usage_Purpose': phone_purpose,
    'Family_Communication': family_comm,
    'Weekend_Usage_Hours': weekend_usage
}])

if st.button("Predict Addiction Level", type="primary"):
    prediction = model.predict(input_data)[0]
    st.success(f"### Predicted Addiction Level: {prediction:.2f} / 10")
    
    if prediction >= 8:
        st.error("⚠️ **High Risk of Phone Addiction.** Consider immediate lifestyle interventions and reducing daily usage hours.")
    elif prediction >= 5:
        st.warning("⚡ **Moderate Risk.** Monitor phone checks and social media time. Encourage more sleep and offline activities.")
    else:
        st.info("✅ **Low Risk.** Healthy digital habits detected.")
