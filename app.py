import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Load from file
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

def transform(input):
    numeric_features = input[['age', 'bmi', 'children']]
    categoric_col = input[['sex', 'smoker', 'region']]
    categoric_features = pd.get_dummies(categoric_col)
    features = pd.merge(numeric_features, categoric_features, left_index=True, right_index=True)

    full_columns_set = ['age', 'bmi', 'children', 'sex_female', 'sex_male','smoker_no',
                        'smoker_yes', 'region_northeast', 'region_northwest',
                        'region_southeast', 'region_southwest']

    features = features.reindex(labels=full_columns_set,axis=1).fillna(0).astype('Int64')

    poly = PolynomialFeatures(degree=3)
    input_transform = poly.fit_transform(features)
    return input_transform

def prediction(model, input_df):
    input_transform = transform(input_df).copy()
    input_df['Prediction'] = abs(model.predict(input_transform)).tolist()
    return input_df


def run():

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict patient hospital charges')
    
    st.title("Insurance Charges Prediction App")

    if add_selectbox == 'Online':

        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = prediction(model, input_df)

        st.write(output)

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            try:
                predictions = prediction(model,data)
                st.write(predictions)
            except:
                st.write('Please check uploaded CSV')
                
if __name__ == '__main__':
    run()