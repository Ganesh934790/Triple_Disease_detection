#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:31:53 2024

@author: user
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/home/user/Downloads/machineLearning/SAVED_MODELS/parkinsons_model.sav', 'rb'))

def parkinsons_predictions(input_data):
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)


    if (prediction[0] == 0):
      print("The Person does not have Parkinsons Disease")

    else:
      print("The Person has Parkinsons")
      
      
def main():
    # giving a title
    st.title('Parkinsons prediction Web App')
    
    
    
    mdvp_fo_hz = st.number_input('MDVP:Fo(Hz)', min_value=0.0)
    mdvp_fhi_hz = st.number_input('MDVP:Fhi(Hz)', min_value=0.0)
    mdvp_flo_hz = st.number_input('MDVP:Flo(Hz)', min_value=0.0)
    mdvp_jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0)
    mdvp_jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0)
    mdvp_rap = st.number_input('MDVP:RAP', min_value=0.0)
    mdvp_ppq = st.number_input('MDVP:PPQ', min_value=0.0)
    jitter_ddp = st.number_input('Jitter:DDP', min_value=0.0)
    mdvp_shimmer = st.number_input('MDVP:Shimmer', min_value=0.0)
    mdvp_shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0)
    shimmer_apq3 = st.number_input('Shimmer:APQ3', min_value=0.0)
    shimmer_apq5 = st.number_input('Shimmer:APQ5', min_value=0.0)
    mdvp_apq = st.number_input('MDVP:APQ', min_value=0.0)
    shimmer_dda = st.number_input('Shimmer:DDA', min_value=0.0)
    nhr = st.number_input('NHR', min_value=0.0)
    hnr = st.number_input('HNR', min_value=0.0)
    status = st.selectbox('Status', ['PD', 'Control'])
    rpde = st.number_input('RPDE', min_value=0.0)
    dfa = st.number_input('DFA', min_value=0.0)
    spread1 = st.number_input('spread1', min_value=0.0)
    spread2 = st.number_input('spread2', min_value=0.0)
    d2 = st.number_input('D2', min_value=0.0)
    ppe = st.number_input('PPE', min_value=0.0)
    
    
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Parkinson Test Result'):
        diagnosis=parkinsons_predictions([mdvp_fo_hz, mdvp_fhi_hz, mdvp_flo_hz, mdvp_jitter_percent, 
                      mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp, 
                      mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, 
                      mdvp_apq, shimmer_dda, nhr, hnr, status, rpde, dfa, 
                      spread1, spread2, d2, ppe])
    
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()