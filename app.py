import streamlit as st
import pandas as pd
from PIL import Image
from config import IMG_PATH, DF_PATH
from model_utils import user_display, outlierHandling, imputingSmokingStatus
from eda_utils import display_visualizations

from sklearn.model_selection import train_test_split

df = pd.read_csv(DF_PATH)
df.drop(columns=['id'], inplace=True)

X = df.drop('stroke' ,axis= 1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= .8, random_state=42)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


st.set_page_config(page_title="Stroke Predictor")

st.sidebar.title("Choose Category")
st.sidebar.divider()

if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.sidebar.button("Home"):
    st.session_state.page = 'home'

if st.sidebar.button("Explore Data"):
    st.session_state.page = 'eda'

if st.sidebar.button("Predict Stroke"):
    st.session_state.page = 'pred'

    
if st.session_state.page == 'eda':
    display_visualizations()

elif st.session_state.page == 'pred':
    user_display(X_train)

elif st.session_state.page == 'home':
    st.header('Health Care Analysis and Stroke Predictor')
    img = Image.open(IMG_PATH)
    st.image(img)
    st.subheader('Introduction')
    st.markdown("Welcome to the Stroke Risk Prediction web application. This tool was developed using Streamlit " \
    "to help users assess the likelihood of experiencing a stroke based on key health and lifestyle indicators.\n " \
    "The model behind this app was trained on a healthcare medical dataset and utilizes machine learning to make predictions. " \
    "By entering information such as age, gender, health conditions (like hypertension or heart disease), and lifestyle factors (such as smoking status and work type), " \
    "the app provides an instant stroke risk prediction.")

    st.markdown("## 📊 About the Dataset")
    st.dataframe(df.head())
    st.write("Shape: ",df.shape)
    st.markdown("There are 5110 Row and 12 Column \n The Dataframe contains about 3 Dtypes:\n- **int64**\n- **float64**\n- **object** which will be turned later into **category**")

    st.markdown("### What are the features in the dataset and what are there dtype")
    for index, col in enumerate(df.columns.tolist()):
        st.markdown(f"{index+1}- {col} (**{df[col].dtype}**)")
    
    st.markdown("### Values of categorical features and there percentage")
    cat_features = df.select_dtypes('object')
    for col in cat_features:
        st.write(f"Values in {col}:")
        st.write(df[col].value_counts(normalize=True)*100)
    
    st.markdown('### Facts about Numeric features')
    st.write(df.describe())

    st.markdown("### Preprocessing Used:\n" \
    "- Handled missing values.\n" \
    "- Encoded categorical variables.\n" \
    "- Scaled numerical features.\n" \
    "- Addressed class imbalance using techniques such as oversampling or SMOTE.\n")

    st.markdown("### 🧠 Model Overview")
    st.markdown("To predict stroke risk, this application uses a Logistic Regression model — a widely-used, interpretable algorithm for binary classification tasks.")
    st.markdown(
        """
        #### ✅ Why Logistic Regression?

        - **Interpretable**: Each feature’s impact on the prediction is easy to understand, making it ideal for health-related applications.  
        - **Efficient**: Performs well with structured data and doesn’t require heavy computational resources.  
        - **Probabilistic Output**: It estimates the probability of stroke, not just a binary prediction, giving a clearer picture of risk.
        """
    )



    
