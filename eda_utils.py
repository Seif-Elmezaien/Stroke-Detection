import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st
from config import DF_PATH, binary_col



def visualization_preprocessing(path):
    """
    Load and Preprocess the data for visualization

    Parameters:
    - path (string): Path to the CSV File

    Returns:
    - pd.Dataframe: An updated DataFrame to visualize on
    - list : A list of categorical names
    - list : A list of numerical names
    """
    df = pd.read_csv(path)
    df.drop(columns=['id'], inplace=True)

    for col in binary_col:
        df[col] = df[col].map(lambda value: col if value == 1 else "No " + col)

    cat_col = df.select_dtypes('object').columns
    numeric_col = df.select_dtypes('number').columns

    return df, cat_col, numeric_col



def get_percent(x, y):
    """
    Calculate the percentage of x relative to y.

    Parameters:
    - x (float or int): The part value.
    - y (float or int): The total value.

    Returns:
    - float: The percentage value rounded to 2 decimal places.
    """
    return round(x / y * 100, 2)



def stroke_histplot(df):
    """
    Takes from the DataFrame the number of people who has stroke
    and the number of people who doesn't and get the precentage of how many people got stroke
    also plot a histogram of the stroke

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization

    Returns:
    - Streamlit Plot element : Plot of the stroke
    - Streamlit message element: A message displayed to the user contains the stroke and no_stroke and its percentage
    """
    no_stroke_count = df['stroke'].value_counts()[0]
    stroke_count = df['stroke'].value_counts()[1]
    stroke_precentage = round(df['stroke'].value_counts(normalize=True)[1] * 100, 2)

    fig = px.histogram(df, 'stroke', color='stroke')
    st.plotly_chart(fig)

    st.markdown(f"We can see that there is a big gap between people that got stroke which are about **{stroke_count}** sample and people who doesn't which are about **{no_stroke_count}**. That is called Imbalanced Classification where the model prediction may be affected by the lack of people with stroke." \
    "So we will need to oversample the data for better model classification")
    st.markdown(f"Samples with stroke represent about {stroke_precentage}%")



def distribution_all_categorical(df, cat_col):
    """
    Makes a count plot of all the categorical data in the DataFrame

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization
    - cat_col (list): List of all the categorical features

    Returns:
    - Streamlit Plot element : Count plot of the categroical features
    - Streamlit message element: A message displayed to the user contains the info of the plot
    """
    fig =plt.figure(figsize=(20,14))
    for i,col in enumerate(cat_col[:-1]):
        plt.subplot(2,4,i+1)
        sns.countplot(data= df, x=col, palette='rocket')
        plt.title(f'Count plot of {col}', fontsize = 18)
        plt.xticks(rotation = 45, fontsize = 15)
        plt.xlabel(f'{col}',fontsize = 16)
    plt.tight_layout()
    
    st.pyplot(fig)
    st.markdown("We can see that: \n" \
    "- Females are more Than Males\n" \
    "- People with no hypretension are more than people with\n" \
    "- People with no heart diesese are more than preople with\n" \
    "- Married are more than none Married\n" \
    "- Most people work Privte\n" \
    "- Most of them are Urban but not by much\n" \
    "- Most of them never Smoked")



def stroke_between_gender(df):
    """
    Make a plot between the gender with stroke
    first takes all the samples with stroke then do the plot
    then it takes the count of males and females with stroke and takes the total count of each one
    the it gets the precent of how many males got stroke to the total males and how many females got stroke to the total females

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization

    Returns:
    - Streamlit Plot element : Plot of the gender with stroke
    - Streamlit Plot element : Plot of the gender with and without stroke
    - Streamlit message element: A message displayed to the user contains the gender and its percentage
    """
    df_stroke = df[df['stroke'] == 'stroke']
    fig = px.histogram(df_stroke, x="gender", color="gender" ,title="Distrubtion of gender that has stroke")
    st.plotly_chart(fig)

    st.markdown("We can see clearly that females got more stroke than males. **BUT** the overall females in the dataset are more than males so we should see the distriburion of each one")

    female_stroke_count = df_stroke['gender'].value_counts()[0]
    male_stroke_count = df_stroke['gender'].value_counts()[1]
    total_female_count = df['gender'].value_counts()[0]
    total_male_count = df['gender'].value_counts()[1]

    female_stroke_pecent =  get_percent(female_stroke_count, total_female_count)
    male_stroke_pecent   =  get_percent(male_stroke_count, total_male_count)

    fig = px.histogram(df, x="gender", color="stroke" ,title="Distrubtion of gender that has stroke", barmode='group')
    st.plotly_chart(fig)

    st.markdown(f"We can see now that from **{total_male_count}** Male there are only **{male_stroke_count}** of them got stroke which are about **{male_stroke_pecent}%** of the Males")
    st.markdown(f"We can see now that from **{total_female_count}** Female there are only **{female_stroke_count}** of them got stroke which are about **{female_stroke_pecent}%** of the Females")
    st.markdown("That means that males get more stroke from the total males in the dataset than the females get from the total females")



def stroke_with_heart_disease(df):
    """
    Makes a plot between having a heart_disease and having stroke
    first making a df of people that got heart disease
    then the count of people that got heart disease and stroke
    than the total count of people that got heart disease
    the get there percentage

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization

    Returns:
    - Streamlit Plot element : Plot of people with heart_disease with respect to stroke
    - Streamlit message element: A message displayed of total heart disease and percentage of people that gets stroke with heart_disease
    """
    df_heart_disease = df[df['heart_disease'] == 'heart_disease']
    heart_disease_stroke_count = df_heart_disease['stroke'].value_counts()[1] # -> People with heart_disease and stroke
    total_heart_disease_count = df['heart_disease'].value_counts()[1] # -> All People with heart_disease

    heart_disease_stroke_persent =  get_percent(heart_disease_stroke_count, total_heart_disease_count)

    fig = px.histogram(df_heart_disease, x="heart_disease", color="stroke", barmode='group', title="Distrubtion of heart_disease that has stroke")
    st.plotly_chart(fig)

    st.markdown(f"From **{total_heart_disease_count}** patient with heart disease there are about **{heart_disease_stroke_count}** that got stroke which means that {heart_disease_stroke_persent}% of them got stroke")



def stroke_with_hypertension(df):
    """
    Makes a plot between having a hypertension and having stroke
    first making a df of people that got hypertension
    then the count of people that got hypertension and stroke
    than the total count of people that got hypertension
    the get there percentage

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization

    Returns:
    - Streamlit Plot element : Plot of people with hypertension with respect to stroke
    - Streamlit message element: A message displayed of total hypertension and percentage of people that gets stroke with hypertension
    """
    df_hypertension = df[df['hypertension'] == 'hypertension']
    hypertension_stroke_count = df_hypertension['stroke'].value_counts()[1] # -> People with heart_disease and stroke
    total_hypertension_count = df['heart_disease'].value_counts()[1] # -> All People with heart_disease

    hypertension_stroke_persent =  get_percent(hypertension_stroke_count, total_hypertension_count)

    fig = px.histogram(df_hypertension, x="hypertension", color="stroke", barmode='group', title="Distrubtion of hypertension that has stroke")
    st.plotly_chart(fig)

    st.markdown(f"From **{total_hypertension_count}** patient with heart disease there are about **{hypertension_stroke_count}** that got stroke which means that {hypertension_stroke_persent}% of them got stroke")



def stroke_with_worktype(df):
    """
    Make a plot between the worktypes with stroke
    first takes all the samples with stroke then do the plot
    then it takes the count of worktypes with stroke and takes the total count of each one
    the it gets the precent of how many worktype got stroke to the total worktype

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization

    Returns:
    - Streamlit Plot element : Plot of the worktypes with stroke
    - Streamlit message element : Info about plot
    - Streamlit message element: A message displayed showing the worktypes and its percentage
    """
    df_stroke = df[df['stroke'] == 'stroke']
    fig = px.histogram(df_stroke, 'work_type', color='stroke', barmode='group', title="Work_types that got stroke")
    st.plotly_chart(fig)

    st.markdown("People with private worktype has more chance of getting stroke than others **BUT** we need to check like we did in the gender feature")
    private_stroke_count = df_stroke['work_type'].value_counts()[0]
    Self_employed_stroke_count = df_stroke['work_type'].value_counts()[1]
    Govt_job_stroke_count = df_stroke['work_type'].value_counts()[2]
    children_stroke_count = df_stroke['work_type'].value_counts()[3]

    total_private = df['work_type'].value_counts()[0]
    total_Self_employed = df['work_type'].value_counts()[1]
    total_Govt_job = df['work_type'].value_counts()[3]
    total_children = df['work_type'].value_counts()[2]

    private_percentage = get_percent(private_stroke_count, total_private)
    Self_employed_percentage = get_percent(Self_employed_stroke_count, total_Self_employed)
    Govt_job_percentage = get_percent(Govt_job_stroke_count, total_Govt_job)
    children_percentage = get_percent(children_stroke_count, total_children)

    st.markdown("We will notice that:\n" \
    f"- People who works private got about {private_percentage}%\n"\
    f"- People who are Self Employed got about {Self_employed_percentage}%\n"\
    f"- People who are working in Governament got about {Govt_job_percentage}%\n"\
    f"- Childern got about {children_percentage}%\n"\
    f"- People Who doesn't work didn't get stroke at all")



def distribution_all_numerical(df, numeric_col):
    """
    Makes a kdeplot of all the numerical data in the DataFrame

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization
    - numeric_col (list): List of all the numerical features

    Returns:
    - Streamlit Plot element : Kde plot of the numerical features
    - Streamlit message element: A message contains the info of the plot
    """
    fig = plt.figure(figsize=(50,20))
    for i, col in enumerate(numeric_col):
        plt.subplot(1,3,i+1)
        sns.kdeplot(data= df, x= col, color='purple')
        plt.title(f'kdeplot of {col}', fontsize = 40)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.xlabel(col, fontsize = 35)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("We can notice that the **average_glucose_level** and **bmi** are both right skewed")



def box_plot(df, col):
    """
    Makes a boxplot for the input col

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization
    - col (string): The name of the column to visualize

    Returns:
    - Streamlit message element: A message contains the Name
    - Streamlit Plot element : Boxplot of the feature
    """
    st.markdown(f"Boxplot of {col}")
    fig = px.box(df, col)
    st.plotly_chart(fig)



def age_stroke_plot(df):
    """
    Makes a histogram of age with color of stroke

    Parameters:
    - df (pd.DataFrame): DataFrame for the visualization

    Returns:
    - Streamlit Plot element : Histogram of age with color of stroke
    - Streamlit message element: A message contains the info about the plot
    """
    fig = px.histogram(df, x='age', nbins=90, color='stroke', title='Age Distribution by Stroke Status')
    st.plotly_chart(fig)

    st.markdown("We can notice that stroke started appearing in a certain age which means there is a relation between them")

    

df, cat_col, numeric_col = visualization_preprocessing(DF_PATH)

def display_visualizations():
    """
    Display the Plots for the user
    """
    st.title('Exploratory Data Analysis 📊')

    st.markdown("### How many people in the dataset had a stroke?")
    stroke_histplot(df)
    st.divider()
    
    st.markdown('### The Distribution of all the Categorical Features')
    distribution_all_categorical(df, cat_col)
    st.divider()

    st.markdown("### Who does get more Stroke between Male and Females?")
    stroke_between_gender(df)
    st.divider()

    st.markdown("### Does having a heart disease inc the prob of having stroke")
    stroke_with_heart_disease(df)
    st.divider()

    st.markdown("### Does having hypertension inc the prob of having stroke")
    stroke_with_hypertension(df)
    st.divider()

    st.markdown("### Does Worktype has an effect on stroke?")
    stroke_with_worktype(df)
    st.divider()

    st.markdown("### Distribution of the Numerical Features")
    distribution_all_numerical(df, numeric_col)
    st.divider()

    st.markdown("### Boxblot of Numeric features")
    box_plot(df, 'age')
    box_plot(df, 'avg_glucose_level')
    box_plot(df, 'bmi')
    st.markdown("**average_glucose_level** and **bmi** both  got outliers")
    st.divider()

    st.markdown("### Does age affect stroke?")
    age_stroke_plot(df)