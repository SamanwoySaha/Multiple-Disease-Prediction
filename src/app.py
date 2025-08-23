import pickle
import os
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading saved models
diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
diabetes_scaler = pickle.load(open('models/diabetes_scaler.sav', 'rb'))

parkinsons_model = pickle.load(open('models/parkinsons_model.sav', 'rb'))
parkinsons_scaler = pickle.load(open('models/parkinsons_scaler.sav', 'rb'))

heart_disease_model = pickle.load(open('models/heart_disease_model.sav', 'rb'))
heart_disease_scaler = pickle.load(open('models/heart_disease_scaler.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
  selected = option_menu('Multiple Disease Prediction System using ML', 
    [
      'Diabetes Prediction',
      'Parkinsons Disease Prediction',
      'Heart Disease Prediction'
    ], 
    menu_icon='hospital-fill',
    icons=['bandaid', 'person', 'heart-pulse'], 
    default_index=0
  )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
  # page title
  st.title('Diabetes Prediction')

  # getting input from user
  col1, col2, col3 = st.columns(3)

  with col1:
    Pregnancies = st.text_input('Number of Pregnancies')
    SkinThickness = st.text_input('Skin Thickness')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')

  with col2:
    Glucose = st.text_input('Glucose Level')
    Insulin = st.text_input('Insulin Level')
    Age = st.text_input('Age')
    
  with col3:
    BloodPressure = st.text_input('Blood Pressure')
    BMI = st.text_input('BMI')
  
  # prediction
  diabetesDiagnosis = ''

  if st.button('Diabetes Test Result'):
    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    user_input = [float(x) for x in user_input]

    user_input = np.array(user_input).reshape(1, -1)

    user_input = diabetes_scaler.transform(user_input)

    diabetesPrediction = diabetes_model.predict(user_input)

    if diabetesPrediction[0] == 1:
      diabetesDiagnosis = 'The person is Diabetic'
    else:
      diabetesDiagnosis = 'The person has no Diabetes'

  st.success(diabetesDiagnosis)

# Parkinsons Disease Prediction Page
if selected == 'Parkinsons Disease Prediction':
  # page title
  st.title('Parkinsons Disease Prediction')

  # getting input from user
  col1, col2, col3 = st.columns(3)

  with col1:
    Fo = st.text_input('Average vocal fundamental frequency')
    Fhi = st.text_input('Maximum vocal fundamental frequency')
    Flo = st.text_input('Minimum vocal fundamental frequency')
    Jitter = st.text_input('Mean Declination of Vocal Pitch (Jitter %)')
    JitterAbs = st.text_input('Mean Declination of Vocal Pitch (Jitter absolute)')
    D2 = st.text_input('Correlation Dimension')
    PPE = st.text_input('Pitch Period Entropy')
    DFA = st.text_input('Detrended Fluctuation Analysis')

  with col2:
    RAP = st.text_input('Rapid eye movement (REM) sleep latency')
    PPQ = st.text_input('Mean Declination of Vocal Pitch (Five-point Period Perturbation Quotient)')
    JitterDDP = st.text_input('Jitter (Average Absolute Difference between the Jittered data and the Smoothed data)')
    Shimmer = st.text_input('Shimmer')
    ShimmerdB = st.text_input('Shimmer in decibels')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')    

  with col3:
    ShimmerAPQ3 = st.text_input('Shimmer (Amplitude Perturbation Quotient 3)')
    ShimmerAPQ5 = st.text_input('Shimmer (Amplitude Perturbation Quotient 5)')
    MDVPAPQ = st.text_input('Mean Declination of Vocal Pitch (Amplitude Perturbation Quotient)')
    ShimmerDDA = st.text_input('Shimmer (Average Absolute Difference between the Shimmered data and the Smoothed data)')
    NHR = st.text_input('Noise-to-Harmonics Ratio')
    HNR = st.text_input('Harmonics-to-Noise Ratio')
    RPDE = st.text_input('Recurrence Period Density Entropy')
    
  # prediction
  parkinsonsDiagnosis = ''

  if st.button('Parkinsons Test Result'):
    user_input = [Fo, Fhi, Flo, Jitter, JitterAbs, RAP, PPQ, JitterDDP, Shimmer, ShimmerdB, ShimmerAPQ3, ShimmerAPQ5, MDVPAPQ, ShimmerDDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

    user_input = [float(x) for x in user_input]

    user_input = np.array(user_input).reshape(1, -1)

    user_input = parkinsons_scaler.transform(user_input)

    parkinsonsPrediction = parkinsons_model.predict(user_input)

    if parkinsonsPrediction[0] == 1:
      parkinsonsDiagnosis = 'The person has Parkinsons Disease'
    else:
      parkinsonsDiagnosis = 'The person has no Parkinsons Disease'

  st.success(parkinsonsDiagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
  # page title
  st.title('Heart Disease Prediction')

  # getting input from user
  col1, col2, col3 = st.columns(3)

  with col1:
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    cp = st.text_input('Systolic Blood Pressure')
    trestbps = st.text_input('Diastolic Blood Pressure')
    chol = st.text_input('Total Cholesterol')
    
  with col2:
    slope = st.text_input('Slope')
    ca = st.text_input('Number of Major Vessels')    
    thal = st.text_input('Thalassemia')
    oldpeak = st.text_input('ST Segment Elevation Myocardial Infarction')    

  with col3:
    fbs = st.text_input('Fasting Blood Sugar')
    restecg = st.text_input('Resting Electrocardiographic Results')
    thalach = st.text_input('Maximum Heart Rate Achieved')
    exang = st.text_input('Exercise Induced Angina')
    
  # prediction
  heartDiseaseDiagnosis = ''

  if st.button('Heart Disease Test Result'):
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    user_input = [float(x) for x in user_input]

    user_input = np.array(user_input).reshape(1, -1)

    user_input = heart_disease_scaler.transform(user_input)

    heartDiseasePrediction = heart_disease_model.predict(user_input)

    if heartDiseasePrediction[0] == 1:
      heartDiseaseDiagnosis = 'The person has Heart Disease'
    else:
      heartDiseaseDiagnosis = 'The person has no Heart Disease'

  st.success(heartDiseaseDiagnosis)