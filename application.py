import streamlit as st
import pandas as pd
import pickle

# Load the machine learning model
with open('artifacts\model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the preprocessing pipeline
with open('artifacts\preprocessor.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

# Define the function to predict math score
def predict_math_score(input_data):
    preprocessed_data = preprocessing_pipeline.transform(input_data)
    prediction = model.predict(preprocessed_data)
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('Student Math Score Predictor')

    # Input form for user to enter parameters
    st.sidebar.header('Input Parameters')
    gender = st.sidebar.radio("Gender", ['female', 'male'])
    race_ethnicity = st.sidebar.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_level_of_education = st.sidebar.selectbox('Parental Level of Education', ["bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school or some high school'])
    lunch = st.sidebar.selectbox('Lunch', ['standard', 'free/reduced'])
    test_preparation_course = st.sidebar.selectbox('Test Preparation Course', ['none', 'completed'])
    reading_score = st.sidebar.number_input('Reading Score',min_value=0, max_value=100)
    writing_score = st.sidebar.number_input('Writing Score', min_value=0, max_value=100)

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'race_ethnicity': [race_ethnicity],
        'parental_level_of_education': [parental_level_of_education],
        'lunch': [lunch],
        'test_preparation_course': [test_preparation_course],
        'reading_score': [reading_score],
        'writing_score': [writing_score],
    })

    # Display input data
    st.subheader('Input Data')
    st.write(input_data)

    # Predict math score
    if st.sidebar.button('Predict'):
        prediction = predict_math_score(input_data)
        st.subheader('Predicted Math Score')
        st.write(prediction)

if __name__ == '__main__':
    main()