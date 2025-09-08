import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from config import MODEL_PATH, gender_features, residence_features, work_type_features, smoke_features


# Function For the Pipline
def outlierHandling(X):
    num_features = ['age', 'avg_glucose_level', 'bmi']
    X = pd.DataFrame(X, columns=num_features)
    for col in num_features:
        Q1 = X[col].quantile(.25)
        Q3 = X[col].quantile(.75)
        IQR = Q3 - Q1

        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        X[col] = X[col].apply(lambda x : lower_fence if x < lower_fence else upper_fence if x > upper_fence else x)
    return X

def imputingSmokingStatus(X):
    cat_features = X.select_dtypes('object').columns
    X = pd.DataFrame(X, columns=cat_features)
    X['smoking_status'].fillna('Unknown', inplace = True)
    return X


def df_maker(gender, age, hypertension, heart_disease, married, work_type, residence_type, avr_glucose_level, bmi, smoking_state):
    """
    Takes the user Inputs and turn it into a dataframe for the model

    Parameter:
    - gender (string): Gender of the user(Male, Female).
    - age (float): Age of the user.
    - hypertension (boolean): If the user got hypertension or not.
    - heart_disease (boolean): If the user got heart disease or not.
    - married (string): If the user is married or not (Yes, No).
    - work_type (string): Which type of work does the user works (Private, Self Employed, Child, Government Job, Never Worked).
    - residence_type (string): What type of residence does the user live in (Urban, Rural).
    - avr_glucose_level (float): Average glucose level of the user.
    - bmi (float): Bmi of the user.
    - smoking_state (string): The smoking state of the user (Never Smoked, Formerly Smokes, Smokes).

    Returns:
    - pd.DataFrame: A dataframe of all the prev parameters.
    """

    df = pd.DataFrame(data= {
        'gender': [gender],
        'age': [age],
        'hypertension': [int(hypertension)],
        'heart_disease': [int(heart_disease)],
        'ever_married': [married],
        'work_type': [work_type_features[work_type]],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avr_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoke_features[smoking_state]],
    })

    return df


# loading Pipline -> A lambda function that loads the model pipline
load_pipline = lambda path: joblib.load(path)


# model prediction
def model_pred(user_input, model_path):
    """
    Takes the user inputs(Dataframe) and the path to model to load model and then predict the inputs and show it to user

    Parameter:
    - X (pd.DataFrame): Dataframe that contains the user inputs.
    - model_path (string): Path to the model to load it.

    Returns:
    - Streamlit message element: A success or warning message displayed to the user based on the model prediction.
    """

    # Loading Model
    model = load_pipline(model_path)

    #prediction
    prediction = model.predict(user_input)
    stroke_prop = model.predict_proba(user_input)[0][1]
    
    return st.success(f"✅ The Model Predict You Are safe With a {round(stroke_prop * 100, 2)}% Probabilty of having Stroke") if prediction == 0 else st.warning(f"⚠️ The Model Recommends checking with a Doctor soon for your own Safety \n\n You have {round(stroke_prop * 100, 2)}% Probabilty of having Stroke.")


def shap_preprocesser(preprocessor, user_input):
    """
    Preprocess the data for the visualization by getting the names preprocessed data.
    Get the user input preprocessed and then turn them into DataFrame for visualization

    Parameter:
    - preprocessor: The preprocesser used in the pipeline.
    - user_input : A single-row DataFrame representing the user's input for explanation.

    Returns:
    - X_transformed_df (pd.DataFrame) : DataFrame of the Target and preprocessed for the Visualization
    """
    # Features names
    num_features = preprocessor['num']['imputer'].get_feature_names_out().tolist()
    bin_features = preprocessor['bin'].get_feature_names_out().tolist()
    cat_features = preprocessor['cat']['encoder'].get_feature_names_out().tolist()

    feature_names = num_features + bin_features + cat_features

    # Transformed input
    X_transformed = preprocessor.transform(user_input)

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    return X_transformed_df


def shap_initialization(model, X_train, user_input):
    """
    Initializes the SHAP explainer for a linear model, computes SHAP values for the given user input,
    and returns the SHAP values.

    Parameters:
    - model: The trained linear model.
    - X_train (pd.DataFrame): The training data used to initialize the SHAP explainer.
    - user_input (pd.DataFrame): A single-row DataFrame representing the user's input for explanation.

    Returns:
    - shap_values: The computed SHAP values for the user input.
    """
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer(user_input)
    return shap_values


def shap_visualization(model_path, X_train, user_input):
    """
    Load the Preprocessor and Model from the pripline than preprocess input and show the visualization
    of the SHAP values

    Parameter:
    - model_path (string): Path to load the model
    - X_train (pd.DataFrame): Dataframe that the model trained on
    - user_input (pd.DataFrame): Dataframe that contains the user inputs.


    Returns:
    - Streamlit plot element: Plots with the info of how each feature affected the prediction.
    """
    pipeline = load_pipline(model_path)
    model = pipeline['model']
    preprocessor = pipeline['preprocess']

    X_train = shap_preprocesser(preprocessor, X_train.sample(100, random_state = 42))
    user_input = shap_preprocesser(preprocessor, user_input)

    shap_values = shap_initialization(model, X_train, user_input)

    st.markdown("This is how each feature affected the prediction")
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.markdown("### How much does each feature affect model in general?")
    st.markdown("Mean Absloute SHAP")

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.markdown("### How much does each feature affect model by value?")
    st.markdown("Beeswarm")

    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()


def user_display(X_train):
    """
    Displays the UI for the user to make his prediction.

    Parameter:
    - X_train (pd.DataFrame): Dataframe that contains the sample the model trained in
    """
    
    st.title("Stroke Prediction 🩺 ")
    st.markdown("### Please fill all the info for prediction")

    gender = st.selectbox("Gender", gender_features)
    age = st.number_input("Age (Must be between 1 and 100)")
    married = st.selectbox("Are You Married?", ["Yes", "No"])
    work_type = st.selectbox("What is your Work type", work_type_features.keys())
    residence = st.select_slider('Residence', residence_features)
    avr_glucose_level = st.number_input("Average Glucose Level (Must be between 50, 300)")
    bmi = st.number_input('BMI (Must be between 10, 100)')
    smoking_state = st.selectbox("Do You Smoke?", smoke_features.keys())
    hypertention = st.checkbox('Do you have Hypertension')
    heart_disease = st.checkbox('Do you have any heart disease?')

    check = False

    check = (age >= 1 and age <= 100) and (bmi >= 10 and bmi <= 100) and (avr_glucose_level >= 50 and avr_glucose_level <= 300)

    if st.button('Submit'):
        if check:
            try:
                user_input = df_maker(gender, age, hypertention, heart_disease, married, work_type, residence, avr_glucose_level, bmi, smoking_state)
                model_pred(user_input, MODEL_PATH)

                st.markdown("### Why did the model predict that?")
                shap_visualization(MODEL_PATH, X_train, user_input)
            except Exception as e:
                st.error(f'Something went wrong {e}')
        else:
            st.error('Cannot Run Until All Inputs are Correct')