import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_california_housing

# Load model
model = joblib.load("house_price_model.pkl")

# Page config
st.set_page_config(page_title="🏠 California House Price Predictor", layout="wide")

# Background styling
page_bg_img = '''
<style>
body {
background-image: linear-gradient(135deg, #e0f7fa, #fffde7);
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🏠 California House Price Predictor")
st.write("### Use the sliders to input features and predict house price.")

# Sidebar Inputs
st.sidebar.header("Input Features")

MedInc = st.sidebar.slider("Median Income (10k $)", 0.5, 15.0, 3.0)
HouseAge = st.sidebar.slider("House Age (years)", 1, 50, 20)
AveRooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Population", 100, 40000, 1000)
AveOccup = st.sidebar.slider("Average Occupants", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.0)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -120.0)

# Prediction
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
predicted_price = model.predict(input_data)[0] * 100000

st.subheader("Predicted House Price:")
st.success(f"💲 ${predicted_price:,.2f}")

# Save predictions
if not os.path.exists("predictions.csv"):
    pd.DataFrame(columns=['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude','Price']).to_csv("predictions.csv", index=False)

new_entry = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, predicted_price]],
                         columns=['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude','Price'])
new_entry.to_csv("predictions.csv", mode='a', header=False, index=False)

# Show prediction logs
if st.checkbox("Show Prediction Log"):
    log = pd.read_csv("predictions.csv")
    st.dataframe(log.tail(10))

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Price'] = data.target
    plt.figure(figsize=(10,7))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
