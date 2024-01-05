import joblib
import streamlit as st
import time
import pandas as pd

# MODEL
model = joblib.load("model/model.joblib")

# STREAMLIT APP
st.set_page_config(page_title="Hungarian Heart Disease Classification", page_icon=":pencil2:")

st.title("Hungarian Heart Disease Classification")
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
  st.header("Single Prediction")
  st.markdown("Enter the values to predict heart disease.")
  st.write("")
  
  col1, col2 = st.columns(2)
  with col1:
    age = st.number_input(label="Age", min_value=0, max_value=100, value=0, step=1)
    st.markdown("<small>:orange[**Min**] value: :orange[**0**], :red[**Max**] value: :red[**100**]</small>", unsafe_allow_html=True)
  with col2:
    sex = st.selectbox(label="Sex", options=["Male", "Female"])
    sex = 1 if sex == "Male" else 0

  col3, col4 = st.columns(2)
  with col3:
    cp = st.selectbox(label="Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    cp = 1 if cp == "Atypical Angina" else 2 if cp == "Non-anginal Pain" else 3 if cp == "Asymptomatic" else 0
  with col4:
    trestbps = st.number_input(label="Resting Blood Pressure", min_value=92.0, max_value=200.0, value=92.0, step=0.01)
    st.markdown("<small>:orange[**Min**] value: :orange[**92.0**], :red[**Max**] value: :red[**200.0**]</small>", unsafe_allow_html=True)

  col5, col6 = st.columns(2)
  with col5:
    chol = st.number_input(label="Serum Cholestoral (in mg/dl)", min_value=85.0, max_value=603.0, value=85.0, step=0.01)
    st.markdown("<small>:orange[**Min**] value: :orange[**85.0**], :red[**Max**] value: :red[**603.0**]</small>", unsafe_allow_html=True)
  with col6:
    fbs = st.selectbox(label="Fasting Blood Sugar", options=["True", "False"])
    fbs = 1 if fbs == "True" else 0

  col7, col8 = st.columns(2)
  with col7:
    restecg = st.selectbox(label="Resting Electrocardiographic Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    restecg = 1 if restecg == "ST-T Wave Abnormality" else 2 if restecg == "Left Ventricular Hypertrophy" else 0
  with col8:
    thalach = st.number_input(label="Maximum Heart Rate Achieved", min_value=82.0, max_value=190.0, value=82.0, step=0.01)
    st.markdown("<small>:orange[**Min**] value: :orange[**82.0**], :red[**Max**] value: :red[**190.0**]</small>", unsafe_allow_html=True)

  col9, col10 = st.columns(2)
  with col9:
    exang = st.selectbox(label="Exercise Induced Angina", options=["Yes", "No"])
    exang = 1 if exang == "Yes" else 0
  with col10:
    oldpeak = st.number_input(label="ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=5.0, value=0.0, step=0.01)
    st.markdown("<small>:orange[**Min**] value: :orange[**0.0**], :red[**Max**] value: :red[**5.0**]</small>", unsafe_allow_html=True)
  
  btn_predict = st.button(label="**Predict**", type="primary")

  if btn_predict:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(101):
      bar.progress(i)
      status_text.text(f"Predicting... {i}%")
      time.sleep(0.01)

      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()
    
    if prediction == 0:
      st.success(":green[**Healthy**]")
    elif prediction == 1:
      st.warning(":orange[**Heart disease level 1**]")
    elif prediction == 2:
      st.error(":orange[**Heart disease level 2**]")
    elif prediction == 3:
      st.error(":red[**Heart disease level 3**]")
    elif prediction == 4:
      st.error(":red[**Heart disease level 4**]")

with tab2:
  st.header("Predict Heart Disease in Batch")
  st.download_button(label="**Download CSV Example**", data=pd.read_csv("data/hungarian.csv").to_csv(index=False), file_name="hungarian.csv", mime="text/csv")

  st.write("")
  uploaded_file = st.file_uploader(label="Upload a CSV file", type=["csv"])

  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    inputs = df

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 50):
      bar.progress(i)
      status_text.text(f"Predicting... {i}%")
      time.sleep(0.01)
    
    results = []
    predictions = model.predict(inputs)
    for prediction in predictions:
      if prediction == 0:
        results.append("Healthy")
      elif prediction == 1:
        results.append("Heart Disease Level 1")
      elif prediction == 2:
        results.append("Heart Disease Level 2")
      elif prediction == 3:
        results.append("Heart Disease Level 3")
      elif prediction == 4:
        results.append("Heart Disease Level 4")

    df["result"] = results

    for i in range(51, 101):
      bar.progress(i)
      status_text.text(f"Predicting... {i}%")
      time.sleep(0.01)

      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()


    st.header("Results")
    st.dataframe(df)
