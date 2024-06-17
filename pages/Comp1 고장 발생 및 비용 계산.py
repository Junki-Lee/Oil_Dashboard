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


st.title("Compnent 1 Analysis")

# 데이터 로드
data = load_data('data/data1_imputed.csv')

# 선택된 변수들 추출
selected_columns = ['AL', 'BA', 'SB', 'CR', 'ZN', 'Y_LABEL']
data = data[selected_columns]

# 특징과 타겟 정의
X = data.drop(columns=['Y_LABEL'])
y = data['Y_LABEL']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = XGBClassifier()
model.fit(X_train, y_train)

# UI 섹션
st.title('건설장비 고장 예측')
st.divider()

st.write("""
건설 장비의 오일 샘플에서 원소 농도를 입력하면 장비의 고장 가능성을 예측할 수 있습니다.
아래에 각 원소의 농도를 입력하세요:
""")

# Input fields for the elements
al_default = data['AL'].mean()
ba_default = data['BA'].mean()
sb_default = data['SB'].mean()
cr_default = data['CR'].mean()
zn_default = data['ZN'].mean()

# Input fields for the elements with dynamic max value
al_max = data['AL'].quantile(0.99)
ba_max = data['BA'].quantile(0.99)
sb_max = data['SB'].quantile(0.99)
cr_max = data['CR'].quantile(0.99)
zn_max = data['ZN'].quantile(0.99)

al = st.slider('알루미늄 (Al)', min_value=0.0, max_value=al_max, step=0.1, value=al_default, key='al_slider')
ba = st.slider('바륨 (Ba)', min_value=0.0, max_value=ba_max, step=0.1, value=ba_default, key='ba_slider')
sb = st.slider('안티모니 (Sb)', min_value=0.0, max_value=sb_max, step=0.1, value=sb_default, key='sb_slider')
cr = st.slider('크롬 (Cr)', min_value=0.0, max_value=cr_max, step=0.1, value=cr_default, key='cr_slider')
zn = st.slider('아연 (Zn)', min_value=0.0, max_value=zn_max, step=0.1, value=zn_default, key='zn_slider')

if st.button('예측'):
    # Prepare the input data
    input_data = np.array([[al, ba, sb, cr, zn]])
    # Make prediction
    prediction_proba = model.predict_proba(input_data) * 100
    prediction = model.predict(input_data)

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
maintenance_cost = st.number_input('부품 교체 비용 (만원)', min_value=0.0, max_value=10000.0, step=0.1, value=300.0)
preventive_maintenance_cost = st.number_input('부품 수리 비용 (만원)', min_value=0.0, max_value=10000.0, step=0.1, value=80.0)
failure_rate = st.number_input('고장 확률 (%)', min_value=0.0, max_value=100.0, step=0.1, value=20.0)

if st.button('비용 절감 계산'):
    # Calculate cost savings
    total_maintenance_cost = maintenance_cost * (failure_rate / 100)
    cost_savings = total_maintenance_cost - preventive_maintenance_cost

    if cost_savings > 0:
        st.write(f"💰 부품을 수리하면 {cost_savings:.2f}만원의 비용을 절감할 수 있습니다.")
    else:
        st.write(f"🔴 부품 수리가 비용 절감에 도움이 되지 않습니다. 고장 시 부품을 교체하세요")