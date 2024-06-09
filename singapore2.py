import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("D:\\project 7\\resale2.pkl", 'rb') as file:
    dt = pickle.load(file)

# Define the encoded values and their corresponding categories
town_names = {
    'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4,
    'BUKIT PANJANG': 5, 'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8,
    'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13,
    'KALLANG/WHAMPOA': 14, 'LIM CHU KANG': 15, 'MARINE PARADE': 16, 'PASIR RIS': 17,
    'PUNGGOL': 18, 'QUEENSTOWN': 19, 'SEMBAWANG': 20, 'SENGKANG': 21, 'SERANGOON': 22,
    'TAMPINES': 23, 'TOA PAYOH': 24, 'WOODLANDS': 25, 'YISHUN': 26
}
room_types = {
    '1 ROOM': 0, '2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4,
    'EXECUTIVE': 5, 'MULTI-GENERATION': 6
}
storey_range_values = {
    '01 TO 03': 0, '04 TO 06': 1, '07 TO 09': 2, '10 TO 12': 3,
    '13 TO 15': 4, '16 TO 18': 5, '19 TO 21': 6, '22 TO 24': 7,
    '25 TO 27': 8, '28 TO 30': 9, '31 TO 33': 10, '34 TO 36': 11,
    '37 TO 39': 12, '40 TO 42': 13, '43 TO 45': 14, '46 TO 48': 15,
    '49 TO 51': 16
}
flat_model_types = {
    '2-ROOM': 0, '3GEN': 1, 'ADJOINED FLAT': 2, 'APARTMENT': 3, 'DBSS': 4,
    'IMPROVED': 5, 'IMPROVED-MAISONETTE': 6, 'MAISONETTE': 7, 'MODEL A': 8,
    'MODEL A-MAISONETTE': 9, 'MODEL A2': 10, 'MULTI GENERATION': 11,
    'NEW GENERATION': 12, 'PREMIUM APARTMENT': 13, 'PREMIUM APARTMENT LOFT': 14,
    'PREMIUM MAISONETTE': 15, 'SIMPLIFIED': 16, 'STANDARD': 17, 'TERRACE': 18,
    'TYPE S1': 19, 'TYPE S2': 20
}

# Define title of app
st.title(':rainbow[Flat Resale Price Prediction]')

# Define user input fields for town and storey range
town = st.selectbox('Town', options=list(town_names.keys()))
storey_range = st.selectbox('Storey Range', options=list(storey_range_values.keys()))

# Define user input fields for flat type and flat model
flat_type = st.selectbox('Flat Type', options=list(room_types.keys()))
flat_model = st.selectbox('Flat Model', options=list(flat_model_types.keys()))

floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=28.0, max_value=307.0, value=50.0)

# Define the year range for lease commence date
min_year = 1990
max_year = 2022

# Define the user field for lease commence date
lease_commence_date = st.number_input('Lease Commencement Year',
                                      min_value=min_year, max_value=max_year,
                                      value=min_year)

# Define the month range for registration month
reg_month = st.selectbox('Registration Month', options=list(range(1, 13)))

# Define the year range for registration year
reg_year = st.number_input('Registration Year', min_value=1990, max_value=2024, value=1990)

# Define the user field for remaining lease in years and months
remaining_lease_year = st.number_input('Remaining Lease (Years)', min_value=0, max_value=97, value=0)
remaining_lease_month = st.selectbox('Remaining Lease (Months)', options=list(range(12)))

# Define the user input field for block number
block = st.number_input('Block Number', min_value=1, max_value=999, value=1)

# Define a function to make predictions
def predict_price(town, flat_type, block, storey_range, floor_area_sqm,
                  flat_model, lease_commence_date, reg_year, reg_month,
                  remaining_lease_year, remaining_lease_month):
    
    # Get the encoded values for flat type and flat model
    encoded_flat_type = room_types[flat_type]
    encoded_flat_model = flat_model_types[flat_model]
    encoded_town = town_names[town]
    encoded_storey_range = storey_range_values[storey_range]
    
    # Log-transform the floor area
    log_floor_area_sqm = np.log(floor_area_sqm) if floor_area_sqm > 0 else 0
    
    # Prepare input data for prediction in the same order as the training data
    input_data = pd.DataFrame({
        'town': [encoded_town],
        'flat_type': [encoded_flat_type],
        'block': [block],
        'storey_range': [encoded_storey_range],
        'floor_area_sqm': [log_floor_area_sqm],
        'flat_model': [encoded_flat_model],
        'lease_commence_date': [lease_commence_date],
        'reg_year': [reg_year],
        'reg_month': [reg_month],
        'remaining_lease_year': [remaining_lease_year],
        'remaining_lease_month': [remaining_lease_month]
    }, columns=['town', 'flat_type', 'block', 'storey_range', 'floor_area_sqm',
                'flat_model', 'lease_commence_date', 'reg_year', 'reg_month',
                'remaining_lease_year', 'remaining_lease_month'])

    # Prediction with RFC
    prediction = dt.predict(input_data)
    
    return prediction

# Button to make predictions
if st.button('Predict'):
    prediction = predict_price(town, flat_type, block, storey_range,
                               floor_area_sqm, flat_model,
                               lease_commence_date, reg_year, reg_month,
                               remaining_lease_year, remaining_lease_month)
    st.write(':green[Predicted Resale Price:]', np.exp(prediction[0]))
