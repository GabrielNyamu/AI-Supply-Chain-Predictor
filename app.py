import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("AI Sales Predictor")

# Made-up data for the demo
data = {'day': [1, 2, 3, 4, 5, 6, 7], 'sales': [10, 12, 11, 15, 25, 40, 35]}
df = pd.DataFrame(data)

# Training the "Brain"
X = df[['day']]
y = df['sales']
model = LinearRegression().fit(X, y)

# The Interface
day = st.slider("Select Day (1=Mon, 7=Sun)", 1, 7, 6)

if st.button("Predict Sales"):
    # This happens instantly
    result = model.predict([[day]])
    st.success(f"AI Forecast: Plan for {round(result[0])} units.")
    st.balloons() # This adds a fun effect to show it worked!