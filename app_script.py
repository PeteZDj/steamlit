import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle

model = pickle.load(open('model_final', 'rb'))
cols=['Provinces', 'category', 'commodity', 'year', 'month']
wholesale_df = pd.read_csv('wholesale_df.csv')
wholesale_encoded = pd.read_csv('wholesale_encoded.csv')

def main():
    # Adding custom CSS to style the app
    st.markdown("""
    <style>
    .stApp {
        background-image: url("finalpic.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    header .css-1595djx {display: none;}  # Hide the default header
    .css-18e3th9 {  # Customize the overall app theme
        color: #ffffff;  # All text colored white
        background-color: #0e1117;  # Background color for containers
    }
    </style>
    """, unsafe_allow_html=True)

    # Logo at the top of the page
    st.image("logo.png", width=100)

    # App title and introduction text
    st.title("Global Harvest Co")
    st.write("Welcome to the GlobalHarvest Co. price prediction app. This app will help you predict prices of various food commodities in various regions in Kenya. Select the Province, the food category, the food commodity, then enter the year and the month. Click on the predict button and the app returns the wholesale price for 1 kilogram.")

    # Styling the selection boxes and buttons
    st.markdown("""
    <style>
    div.stSelectbox {color: black;}  # Custom color for dropdown
    button {border-radius: 20px; font-size: 16px; background-color: #025246; color: white;}
    </style>
    """, unsafe_allow_html=True)

    # Dropdown for Province
    province_name = st.selectbox("Province", wholesale_df['Provinces'].unique())

    filtered_df = wholesale_df[wholesale_df['Provinces'] == province_name]
    food_categories = filtered_df['category'].unique()
    food_category = st.selectbox("Food Category", food_categories)

    filtered_df = filtered_df[filtered_df['category'] == food_category]
    food_commodity = st.selectbox("Food Commodity", filtered_df['commodity'].unique())
    
    # Number input for Year and Month
    year = st.number_input("Year", min_value=2015, max_value=2030, value=2024, step=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=5, step=1)

    if st.button("Predict"):
        prediction_data_dict = {
                'Provinces': province_name,
                'category': food_category,
                'commodity': food_commodity,
                'year': year,
                'month': month
            }
        
        commodities_info = {}
        keys_to_skip = ['year', 'month']
        feature_columns = list(wholesale_encoded.drop(['price/unit(KSH)'], axis=1).columns)

        for feature in feature_columns:
            commodities_info[feature] = 0

        for key, value in prediction_data_dict.items():
            if key not in keys_to_skip:
                element = key + "_" + value
                if element in feature_columns:
                    commodities_info[element] = 1

        commodities_info['year'] = year
        commodities_info['month'] = month

        commodities_data = pd.DataFrame.from_dict(commodities_info, orient='index').T
        predicted = model.predict(commodities_data)

        st.success(f"{round(predicted[0], 2)} KSH")

if __name__=='__main__': 
    main()
