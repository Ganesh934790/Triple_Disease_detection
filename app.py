import os
import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open('/home/user/Downloads/machineLearning/SAVED_MODELS/diabetes_model.sav', 'rb'))

loaded_model = pickle.load(open('/home/user/Downloads/machineLearning/SAVED_MODELS/breast_cancer_model.sav', 'rb'))

parkinsons_model = pickle.load(open('/home/user/Downloads/machineLearning/SAVED_MODELS/parkinsons_model.sav', 'rb'))


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Triple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Breast Cancer Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

#brear_cancer_classification page
if selected=="Breast Cancer Prediction":
    st.title("Breast Cancer Diagnosis Prediction using ML")
    
    # Define columns for input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Input fields for features
    with col1:
        radius_mean = st.text_input('radius_mean')
        compactness_mean = st.text_input('compactness_mean')
        radius_se = st.text_input('radius_se')
        compactness_se = st.text_input('compactness_se')
        radius_worst = st.text_input('radius_worst')
        compactness_worst=st.text_input("compactness_worst")
    with col2:
        texture_mean = st.text_input('texture_mean')
        concavity_mean = st.text_input('concavity_mean')
        texture_se = st.text_input('texture_se')
        concavity_se = st.text_input('concavity_se')
        texture_worst = st.text_input('texture_worst')
        concavity_worst=st.text_input("concavity_worst")
        
    with col3:
        perimeter_mean = st.text_input('perimeter_mean')
        concave_points_mean = st.text_input('concave points_mean')
        perimeter_se = st.text_input('perimeter_se')
        concave_points_se = st.text_input('concave points_se')
        perimeter_worst = st.text_input('perimeter_worst')
        concave_points_worst=st.text_input("concave_points_worst")
    with col4:
        area_mean = st.text_input('area_mean')
        symmetry_mean = st.text_input('symmetry_mean')
        area_se = st.text_input('area_se')
        symmetry_se = st.text_input('symmetry_se')
        area_worst = st.text_input('area_worst')
        symmetry_worst=st.text_input("symmetry_worst")
    with col5:
        smoothness_mean = st.text_input('smoothness_mean')
        fractal_dimension_mean = st.text_input('fractal_dimension_mean')
        smoothness_se = st.text_input('smoothness_se')
        fractal_dimension_se = st.text_input('fractal_dimension_se')
        smoothness_worst = st.text_input('smoothness_worst')
        fractal_dimension_worst=st.text_input("fractal_dimension_worst")
        
    cancer_diagnosis=""
    # Prediction button
    
    if st.button("Diagnosis Result"):
        # Gather user inputs
        user_input = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                      compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                      radius_se, texture_se, perimeter_se, area_se, smoothness_se,
                      compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                      radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                      compactness_worst,concavity_worst	,concave_points_worst,symmetry_worst,fractal_dimension_worst]
        
        # Convert inputs to float
        input_data= [float(x) for x in user_input]
        
       # change the input data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)

       # reshape the numpy array as we are predicting for one datapoint
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
          cancer_diagnosis = 'The Breast cancer is Malignant'

        else:
            cancer_diagnosis = 'The Breast Cancer is Benign'
        
        # Display result
    st.success(cancer_diagnosis)


    
    
    


# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
