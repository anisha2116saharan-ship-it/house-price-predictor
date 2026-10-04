import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Page Setup
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 House Price Prediction Dashboard")
st.markdown("""
This Machine Learning web application estimates residential property values using a 
**Random Forest Regressor** trained on historical housing demographics and structural features.
""")

# 2. Train Model in Memory (Cached so it loads instantly)
@st.cache_resource
def load_and_train():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Features & Target
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal'] * 100000  # Convert to real dollar values (originally in $100k blocks)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train lightweight Random Forest Regressor
    model = RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

with st.spinner("Training Machine Learning Model..."):
    model, feature_names = load_and_train()

# 3. Sidebar Controls for House Features
st.sidebar.header("📐 Property Parameters")

med_inc = st.sidebar.slider("Median Income of Area ($10,000s)", min_value=1.0, max_value=15.0, value=3.87, step=0.1)
house_age = st.sidebar.slider("House Age (Years)", min_value=1, max_value=60, value=28)
ave_rooms = st.sidebar.slider("Average Rooms", min_value=1.0, max_value=10.0, value=5.4, step=0.1)
ave_bedrms = st.sidebar.slider("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.1, step=0.1)
population = st.sidebar.number_input("Neighborhood Population", min_value=100, max_value=30000, value=1425, step=50)
ave_occup = st.sidebar.slider("Average House Occupancy", min_value=1.0, max_value=6.0, value=3.0, step=0.1)
latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.6, step=0.1)
longitude = st.sidebar.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-119.5, step=0.1)

# Format user input into single row DataFrame
input_df = pd.DataFrame([{
    'MedInc': med_inc,
    'HouseAge': house_age,
    'AveRooms': ave_rooms,
    'AveBedrms': ave_bedrms,
    'Population': population,
    'AveOccup': ave_occup,
    'Latitude': latitude,
    'Longitude': longitude
}])

# 4. Display & Prediction
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Property Features")
    display_df = pd.DataFrame({
        "Feature": ["Median Income", "House Age", "Avg Rooms", "Avg Bedrooms", "Local Population", "Avg Occupancy"],
        "Input Value": [f"${med_inc * 10000:,.0f}", f"{house_age} yrs", f"{ave_rooms:.1f}", f"{ave_bedrms:.1f}", f"{population}", f"{ave_occup:.1f}"]
    })
    st.table(display_df)

with col2:
    st.subheader("Estimated Market Valuation")
    if st.button("Predict House Price", type="primary"):
        prediction = model.predict(input_df)[0]
        st.success(f"### Estimated Price: **${prediction:,.2f}**")
        st.info("Valuation calculated using localized geographic coordinates and residential features.")
