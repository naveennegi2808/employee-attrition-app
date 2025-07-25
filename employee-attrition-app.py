import streamlit as st
import joblib
import pandas as pd
import plotly
import plotly.express as px


# Load model 
model = joblib.load('model.pkl')

# App title
st.markdown(
    """
    <h1 style='text-align: center; font-family: Georgia, serif;'>
        Employee Attrition Prediction
    </h1>
    """,
    unsafe_allow_html=True
)   # Input fields
overtime = st.selectbox("OverTime ⌛️", ["Yes", "No"])
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
monthly_income = st.number_input("Monthly Income USD 💵", min_value=1000, max_value=20000, step=500)
age = st.number_input("Age", min_value=18, max_value=60, step=1)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, step=1)
total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, step=1)
distance_from_home = st.number_input("Distance from Home (Kms)", min_value=0, max_value=50, step=1)
job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                         'Manufacturing Director', 'Healthcare Representative',
                                         'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
business_travel = st.selectbox("Business Travel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])

    
# Prediction section — only runs when button is clicked
if st.button("Predict 🧠"):

    input_data = {
        'JobSatisfaction': [job_satisfaction],
        'EnvironmentSatisfaction': [env_satisfaction],
        'MonthlyIncome': [monthly_income],
        'Age': [age],
        'YearsAtCompany': [years_at_company],
        'TotalWorkingYears': [total_working_years],
        'DistanceFromHome': [distance_from_home],
        'OverTime_Yes': [1 if overtime == 'Yes' else 0],
        'JobRole_Human Resources': [1 if job_role == 'Human Resources' else 0],
        'JobRole_Laboratory Technician': [1 if job_role == 'Laboratory Technician' else 0],
        'JobRole_Manager': [1 if job_role == 'Manager' else 0],
        'JobRole_Manufacturing Director': [1 if job_role == 'Manufacturing Director' else 0],
        'JobRole_Research Director': [1 if job_role == 'Research Director' else 0],
        'JobRole_Research Scientist': [1 if job_role == 'Research Scientist' else 0],
        'JobRole_Sales Executive': [1 if job_role == 'Sales Executive' else 0],
        'JobRole_Sales Representative': [1 if job_role == 'Sales Representative' else 0],
        'BusinessTravel_Travel_Frequently': [1 if business_travel == 'Travel_Frequently' else 0],
        'BusinessTravel_Travel_Rarely': [1 if business_travel == 'Travel_Rarely' else 0]
    }

    input_df = pd.DataFrame(input_data)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ High chance of Attrition! (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Low chance of Attrition. (Confidence: {probability:.2f})")

st.subheader("Now let's move on to some Visualisations ! 📈 ")      

#NOW VISUALISATION CODE 

df_fv=pd.read_csv("df_encoded.csv")

# Select plot type
plot_type = st.selectbox("Choose plot type", ["Scatter", "Bar", "Box", "Histogram",])

# Exclude encoded categorical columns like JobRole_*, BusinessTravel_*, OverTime_*
exclude_prefixes = ("JobRole_", "BusinessTravel_", "OverTime_")
all_features = [col for col in df_fv.columns if not col.startswith(exclude_prefixes)]

# Feature selectors
x_feature = st.selectbox("Select X-axis feature", all_features)
y_feature = None

# For all except histogram, require Y feature
if plot_type != "Histogram":
    y_feature = st.selectbox("Select Y-axis feature", all_features)

# Plot generation
if plot_type == "Scatter":
    fig = px.scatter(df_fv, x=x_feature, y=y_feature, color='Attrition', title=f"{x_feature} vs {y_feature}")
elif plot_type == "Bar":
    fig = px.bar(df_fv, x=x_feature, y=y_feature, color='Attrition', barmode="group", title=f"{x_feature} vs {y_feature}")
elif plot_type == "Box":
    fig = px.box(df_fv, x=x_feature, y=y_feature, color='Attrition', title=f"{x_feature} vs {y_feature}")
elif plot_type == "Histogram":
    fig = px.histogram(df_fv, x=x_feature, color='Attrition', title=f"Distribution of {x_feature}")

st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    st.title("WORKPULSE.AI 🤖")
    st.markdown("### 📊 About This App")
    st.write("This app predicts the likelihood of employee attrition based on user-selected features. It is powered by a machine learning model trained on HR analytics data.")
    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.write("""
        - **Model Used**: Balanced Random Forest Classifier  
        - **Target Variable**: Attrition  
    """)    
    with st.expander("### 📌 Instructions"):
        st.write("""
        1. Enter employee details using the input panel.
        2. Click **Predict** to get the attrition likelihood.
        3. Explore data visually in the **Feature Visualization** tab.
         """)
    

