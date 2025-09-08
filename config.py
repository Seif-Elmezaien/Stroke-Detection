# Here where the constants are
import os

BASE_DIR = '/Users/seifeldenelmizayen/Desktop/Aman_Data_Intern/Third Task Better/'

DF_PATH = os.path.join(BASE_DIR, 'dataset', 'healthcare-dataset-stroke-data.csv')
MODEL_PATH = os.path.join(BASE_DIR, "model", "LR_pipline_model.pkl")
IMG_PATH = os.path.join(BASE_DIR, "img", "stroke.jpg")

gender_features = ["Male", "Female"]

residence_features = ["Urban", "Rural"]

work_type_features = {'Private': 'Private',
                  'Self Employed': "Self-employed",
                  'Child': 'children',
                  'Government Job': 'Govt_job',
                  'Never Worked' : "Never_worked"}

smoke_features = {'Never Smoked': 'never smoked',
                  'Formerly Smokes': 'formerly smoked',
                  'Smokes': 'smokes'}

binary_col = ['hypertension', 'heart_disease', 'stroke']
