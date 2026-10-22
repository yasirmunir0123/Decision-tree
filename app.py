
import streamlit as st
import joblib
import pandas as pd

# Website title
st.title("Decision Tree Prediction App")
st.write("Enter flower measurements to predict its class.")

# Load trained model
model = joblib.load("decision_tree_model.pkl")

# User input
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.1, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.1, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.1, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, value=0.2)

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = int(model.predict(input_data)[0])

    classes = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Class: {classes[prediction]}")
