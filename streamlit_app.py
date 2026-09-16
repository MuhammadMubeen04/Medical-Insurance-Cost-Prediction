import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.25);
    }
    .prediction-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.6rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1a365d, #2b6cb0);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        height: 3.1rem;
        font-size: 1.1rem;
        border: none;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2b6cb0, #1a365d);
        color: white;
    }
    /* Clean metric cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1a365d !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("insurance.csv")
    
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df['region'] = df['region'].map({
        'southwest': 0, 'southeast': 1,
        'northwest': 2, 'northeast': 3
    })
    
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = load_and_train_model()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 💰 Insurance Predictor")
    st.markdown("---")
    st.markdown("""
    ### About this App
    Predicts **medical insurance cost** using a Linear Regression model trained on real health data.
    
    ### Features Used
    - Age & Gender  
    - BMI  
    - Number of Children  
    - Smoking Status  
    - Region  
    """)
    st.markdown("---")
    st.caption("AI Internship • Big Brains")

# ==================== MAIN ====================
st.markdown('<p class="main-title">💰 Medical Insurance Cost Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get an estimated insurance cost based on personal and lifestyle factors</p>', unsafe_allow_html=True)

st.markdown("### 📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ["male", "female"])

with col2:
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0, 0.1)
    children = st.slider("Number of Children", 0, 5, 0)

with col3:
    smoker = st.selectbox("Smoking Status", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

st.write("")

if st.button("🔍 Predict Insurance Cost"):
    
    # Encode
    sex_encoded = 1 if sex == "female" else 0
    smoker_encoded = 1 if smoker == "yes" else 0
    region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
    region_encoded = region_map[region]
    
    input_data = [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]]
    prediction = model.predict(input_data)[0]
    
    # Result
    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="margin:0; font-weight:600;">Predicted Medical Insurance Cost</h3>
        <div class="prediction-value">${prediction:,.2f}</div>
        <p style="margin:0; opacity:0.9;">Estimated annual insurance cost</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Summary (Clean version)
    st.markdown("### 📌 Input Summary")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(label="Age", value=f"{age} yrs")
    c2.metric(label="Gender", value=sex.capitalize())
    c3.metric(label="BMI", value=f"{bmi}")
    c4.metric(label="Children", value=str(children))
    c5.metric(label="Smoker", value=smoker.capitalize())
    c6.metric(label="Region", value=region.capitalize())

else:
    st.info("👆 Fill in the details and click **Predict Insurance Cost** to see the result.")

st.markdown("---")
st.caption("Developed for Artificial Intelligence Internship at **Big Brains** | Linear Regression Model")
