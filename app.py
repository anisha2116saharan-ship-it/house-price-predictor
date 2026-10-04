import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Page Setup
st.set_page_config(
    page_title="House Price Predictor (INR)",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 House Price Prediction Dashboard")
st.markdown("""
This Machine Learning web application estimates residential property values in **Indian Rupees (₹)** 
using a **Random Forest Regressor** model.
""")

# Currency Conversion Factor (USD to INR estimate)
USD_TO_INR = 85.0

# Helper function to format INR cleanly in Lakhs or Crores
def format_inr(val):
    if val >= 10000000:
        return f"₹{val / 10000000:.2f} Crores"
    elif val >= 100000:
        return f"₹{val / 100000:.2f} Lakhs"
    else:
        return f"₹{val:,.0f}"

# 2. Train Model in Memory
@st.cache_resource
def load_and_train():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Target value converted to INR
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal'] * 100000 * USD_TO_INR
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

with st.spinner("Training Machine Learning Model..."):
    model = load_and_train()

# 3. Sidebar Controls
st.sidebar.header("📐 Property Parameters")

med_inc = st.sidebar.slider("Locality Annual Income Tier (₹ Lakhs)", min_value=2.0, max_value=30.0, value=8.5, step=0.5)
house_age = st.sidebar.slider("Property Age (Years)", min_value=1, max_value=50, value=12)
ave_rooms = st.sidebar.slider("Average Total Rooms", min_value=1.0, max_value=10.0, value=4.5, step=0.5)
ave_bedrms = st.sidebar.slider("Number of Bedrooms", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
population = st.sidebar.number_input("Locality Population Density", min_value=100, max_value=30000, value=2500, step=100)
ave_occup = st.sidebar.slider("Average Family Size (Occupancy)", min_value=1.0, max_value=6.0, value=4.0, step=0.5)

# Convert user income back to model scale
input_df = pd.DataFrame([{
    'MedInc': (med_inc * 100000) / (10000 * USD_TO_INR),
    'HouseAge': house_age,
    'AveRooms': ave_rooms,
    'AveBedrms': ave_bedrms,
    'Population': population,
    'AveOccup': ave_occup,
    'Latitude': 35.6,
    'Longitude': -119.5
}])

# 4. Display & Prediction
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Property Specs")
    display_df = pd.DataFrame({
        "Parameter": ["Area Income Tier", "Property Age", "Rooms", "Bedrooms", "Locality Population", "Family Size"],
        "Value": [f"₹{med_inc:.1f} Lakhs/yr", f"{house_age} years", f"{ave_rooms:.0f} rooms", f"{ave_bedrms:.0f} BHK", f"{population} residents", f"{ave_occup:.0f} persons"]
    })
    st.table(display_df)

with col2:
    st.subheader("Estimated Market Valuation")
    if st.button("Calculate Property Value", type="primary"):
        prediction = model.predict(input_df)[0]
        formatted_price = format_inr(prediction)
        
        st.success(f"### Estimated Price: **{formatted_price}**")
        st.write(f"*(Exact raw calculation: ₹{prediction:,.2f})*")
        st.info("Valuation calculated based on housing features, family size density, and income demographics.")
