import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
import joblib

model = joblib.load('musicmodel.joblib')
encoder = joblib.load('Encoder.joblib')
label_encoder = joblib.load('labelencoder.joblib')

st.title("Music Genre Predictions")
st.markdown("ML model to predict music genre individual likes based on their age and gender")

st.header("Predictor")

col1, col2 = st.columns(2)
with col1:
    st.text("Age")
    age = st.text_input("Enter Age","")

with col2:
     st.text("Enter Gender")
    #  gender = st.text_input("Enter Male Or Female", "")
     gender = st.selectbox("Select Gender",["","Male","Female"])
# st.button("Predict-Genre")
input_data = pd.DataFrame(data=[[age, gender]], columns=['age', 'gender'])

categorical_column = 'gender'
# if gender in ['Male', 'Female']:
#     input_data[categorical_column] = label_encoder.transform([str(input_data[categorical_column][0])])[0]
# else:
#     st.warning("Please select a valid gender ('Male' or 'Female').")

# input_data[categorical_column] = label_encoder.transform(input_data[categorical_column][0])

if st.button("Predict-Genre"):
    #result = predict(np.array([[age,gender]]))
    if gender in ['Male', 'Female']:
        input_data[categorical_column] = label_encoder.transform([str(input_data[categorical_column][0])])[0]
    else:
        st.warning("Please select a valid Age and Gender.")
    result = predict(input_data)

    predicted_result = encoder.inverse_transform(result)
    st.text(predicted_result[0])