import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from src.utils import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

st.title("Compnent 3 Analysis")

data_path = './data/component3_imputed.csv'

def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def load_data(data_path, selected_features):
    df = pd.read_csv(data_path)
    df = df[selected_features + ['Y_LABEL']]  
    return df

def preprocess_data(df, target_column='Y_LABEL'): 
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_train, X_test, y_train, y_test

def train_student_model(X_train, y_train):
    student_model = XGBClassifier(random_state=42)
    student_model.fit(X_train, y_train)
    return student_model


selected_features = ['AL', 'CA', 'P', 'B', 'S']

# 데이터 로드 및 전처리
df = load_data(data_path, selected_features)
X, y, X_train, X_test, y_train, y_test = preprocess_data(df, target_column='Y_LABEL')

# 학생 모델 학습
student_model = train_student_model(X_train, y_train)

# UI 섹션
st.title('건설장비 고장 예측')
st.divider()

st.write("""
건설 장비의 오일 샘플에서 원소 농도를 입력하면 장비의 고장 가능성을 예측할 수 있습니다.
아래에 각 원소의 농도를 입력하세요:
""")

# Input fields for the elements
al_default = df['AL'].mean()
ca_default = df['CA'].mean()
p_default = df['P'].mean()
b_default = df['B'].mean()
s_default = df['S'].mean()

al = st.number_input('AL', min_value=0.0, max_value=50000.0, step=0.1, value=al_default)
ca = st.number_input('CA', min_value=0.0, max_value=50000.0, step=0.1, value=ca_default)
p = st.number_input('P', min_value=0.0, max_value=50000.0, step=0.1, value=p_default)
b = st.number_input('B', min_value=0.0, max_value=50000.0, step=0.1, value=b_default)
s = st.number_input('S', min_value=0.0, max_value=50000.0, step=0.1, value=s_default)

if st.button('예측'):
    # Prepare the input data
    input_data = np.array([[al, ca, p, b, s]])
    # Make prediction
    prediction_proba = student_model.predict_proba(input_data) * 100
    prediction = student_model.predict(input_data)

    # Display prediction
    st.write(f"고장이 발생할 확률: {prediction_proba[0][1]:.2f} %")

    if prediction[0] == 1:
        st.write("🔴 건설 장비에 고장이 발생할 가능성이 높습니다.")
    else:
        st.write("🟢 건설 장비가 정상 작동할 가능성이 높습니다.")


st.title('유지보수 비용 절감 계산기')
st.divider()

st.write("""
예측 알고리즘을 통해 유지보수 비용 절감 효과를 계산해보세요.
""")

# Input fields for maintenance costs
maintenance_cost = st.number_input('부품 교체 비용 (만원)', min_value=0.0, max_value=10000.0, step=0.1, value=800.0)
preventive_maintenance_cost = st.number_input('부품 수리 비용 (만원)', min_value=0.0, max_value=10000.0, step=0.1, value=350.0)
failure_rate = st.number_input('고장 확률 (%)', min_value=0.0, max_value=100.0, step=0.1)

if st.button('비용 절감 계산'):
    # Calculate cost savings
    total_maintenance_cost = maintenance_cost * (failure_rate / 100)
    cost_savings = total_maintenance_cost - preventive_maintenance_cost

    if cost_savings > 0:
        st.write(f"💰 부품을 수리하면 {cost_savings:.2f}만원의 비용을 절감할 수 있습니다.")
    else:
        st.write(f"🔴 부품 수리가 비용 절감에 도움이 되지 않습니다. 고장 시 부품을 교체하세요")