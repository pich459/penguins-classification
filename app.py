import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.title(body="Penguins Classification")
st.header(body="Input Features")
island = st.selectbox(
    label="Island", 
    options=["Adelie", "Chinstrap", "Gentoo"]
    )

sex = st.radio(
    label="sex",
    options=["Male", "Female"]
    )

bill_length = st.slider(
    label = "bill_length (mm)",
    min_value=30.0, 
    max_value=60.0,
    step=0.1,
)

bill_depth = st.slider(
    label = "bill_depth (mm)",
    min_value=10.0,
    max_value=25.0,
    step=0.1,
)
flipper_length = st.slider(
    label = "flipper_length (mm)",
    min_value=150.0,
    max_value=250.0,
    step=1.0,
)
body_mass = st.slider(
    label = "body_mass (g)",
    min_value=2000.0,
    max_value=7000.0,
    step=100.0,
)   

with open(file="pipeline.pkl", mode="rb") as file:
    model = pickle.load(file=file)
    X_inputs = pd.DataFrame(
        {
            "island": np.array([island], dtype=np.str_),
            "bill_length_mm": np.array([bill_length], dtype=np.float64),
            "bill_depth_mm": np.array([bill_depth], dtype=np.float64),
            "flipper_length_mm": np.array([flipper_length], dtype=np.float64),
            "body_mass_g": np.array([body_mass], dtype=np.float64),
            "sex": np.array([sex], dtype=np.str_),
        },
    )
    y_output = model.predict(X_inputs)
    st.write(y_output)

