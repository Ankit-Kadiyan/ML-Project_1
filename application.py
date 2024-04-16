import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved machine learning model
with open('artifacts\model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess the input data
def preprocess_input(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
    # Perform any necessary preprocessing
    # For example, convert categorical variables to numerical using one-hot encoding
    
    # Gender
    if gender == 'male':
        gender_male = 1
        gender_female = 0
    else:
        gender_male = 0
        gender_female = 1
    
    # Race/Ethnicity
    race_mapping = {'group A': 1, 'group B': 2, 'group C': 3, 'group D': 4, 'group E': 5}
    race_ethnicity_encoded = race_mapping[race_ethnicity.lower()]
    
    # Parental Level of Education
    education_mapping = {"bachelor's degree": 1, 'some college': 2, "master's degree": 3, "associate's degree": 4, 'high school': 5, 'some high school': 6}
    parental_level_of_education_encoded = education_mapping[parental_level_of_education.lower()]
    
    # Lunch
    if lunch == 'standard':
        lunch_standard = 1
    else:
        lunch_standard = 0
    
    # Test Preparation Course
    if test_preparation_course == 'completed':
        test_prep_completed = 1
    else:
        test_prep_completed = 0
    
    # Combine all features into a single numpy array
    features = np.array([gender_male, gender_female, race_ethnicity_encoded, parental_level_of_education_encoded, lunch_standard, test_prep_completed, reading_score, writing_score]).reshape(1, -1)
    
    return features

# Function to predict math score
def predict_math_score(features):
    # Use the loaded model to make predictions
    math_score = model.predict(features)
    return math_score

# Streamlit UI
st.title('Math Score Prediction')

# Inputs
gender = st.selectbox('Gender', ('Male', 'Female'))
race_ethnicity = st.selectbox('Race/Ethnicity', ('group A', 'group B', 'group C', 'group D', 'group E'))
parental_level_of_education = st.selectbox('Parental Level of Education', ("bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school'))
lunch = st.selectbox('Lunch', ('standard', 'free/reduced'))
test_preparation_course = st.selectbox('Test Preparation Course', ('none', 'completed'))
reading_score = st.slider('Reading Score', min_value=0, max_value=100, value=50, step=1)
writing_score = st.slider('Writing Score', min_value=0, max_value=100, value=50, step=1)

# Preprocess inputs
input_features = preprocess_input(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)

# Predict math score
math_score = predict_math_score(input_features)

# Display predicted math score
st.write('Predicted Math Score:', math_score[0])