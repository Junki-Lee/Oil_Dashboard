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

st.title("Compnent 2 Analysis")

data_path = './data/component2_imputed.csv'

# train test 분리
def preprocess_data(df, target_column='Y_LABEL'): 
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_train, X_test, y_train, y_test

# 데이터 로드
data = load_data(data_path)
X, y, X_train, X_test, y_train, y_test = preprocess_data(data, target_column='Y_LABEL')


# 모델 훈련
model = joblib.load('C:/Users/USER/projects/LS_project4/oil_dashboard/models/component2_imputed_model.joblib')

# UI 섹션
st.title('건설장비 고장 예측')
st.divider()

st.write("""
건설 장비의 오일 샘플에서 원소 농도를 입력하면 장비의 고장 가능성을 예측할 수 있습니다.
아래에 각 원소의 농도를 입력하세요:
""")

# Input fields for the elements
ag_default = data['AG'].mean()
al_default = data['AL'].mean()
b_default = data['B'].mean()
ba_default = data['BA'].mean()
be_default = data['BE'].mean()
ca_default = data['CA'].mean()
cd_default = data['CD'].mean()
co_default = data['CO'].mean()
cr_default = data['CR'].mean()
cu_default = data['CU'].mean()
fe_default = data['FE'].mean()
h20_default = data['H2O'].mean()
k_default = data['K'].mean()
li_default = data['LI'].mean()
mg_default = data['MG'].mean()
mn_default = data['MN'].mean()
mo_default = data['MO'].mean()
na_default = data['NA'].mean()
ni_default = data['NI'].mean()
p_default = data['P'].mean()
pb_default = data['PB'].mean()
pqindex_default = data['PQINDEX'].mean()
s_default = data['S'].mean()
sb_default = data['SB'].mean()
si_default = data['SI'].mean()
sn_default = data['SN'].mean()
ti_default = data['TI'].mean()
u100_default = data['U100'].mean()
u75_default = data['U75'].mean()
u50_default = data['U50'].mean()
u25_default = data['U25'].mean()
u20_default = data['U20'].mean()
u14_default = data['U14'].mean()
u6_default = data['U6'].mean()
u4_default = data['U4'].mean()
v_default = data['V'].mean()
v40_default = data['V40'].mean()
zn_default = data['ZN'].mean()

max = 20000

ag = st.number_input('AG', min_value=0.0, max_value=10000.0, step=0.1, value=ag_default)
al = st.number_input('AL', min_value=0.0, max_value=10000.0, step=0.1, value=al_default)
b = st.number_input('B', min_value=0.0, max_value=10000.0, step=0.1, value=b_default)
ba = st.number_input('BA', min_value=0.0, max_value=10000.0, step=0.1, value=ba_default)
be = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=be_default)
ca = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=ca_default)
cd= st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=cd_default)
co = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=co_default)
cr = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=cr_default)
cu = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=cu_default)
fe = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=fe_default)
h20 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=h20_default)
k = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=k_default)
li = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=li_default)
mg = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=mg_default)
mn = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=mn_default)
mo = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=mo_default)
na = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=na_default)
ni = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=ni_default)
p = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=p_default)
pb = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=pb_default)
pqindex = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=pqindex_default)
s = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=s_default)
sb = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=sb_default)
si = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=si_default)
sn = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=sn_default)
ti = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=ti_default)
u100 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=u100_default)
u75 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=u75_default)
u50 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=u50_default)
u25 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=u25_default)
u20 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=u20_default)
u14 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=u14_default)
u6 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=u6_default)
u4 = st.number_input('BE', min_value=0.0, max_value=20000.0, step=0.1, value=u4_default)
v = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=v_default)
v40 = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=v40_default)
zn = st.number_input('BE', min_value=0.0, max_value=10000.0, step=0.1, value=zn_default)

if st.button('예측'):
    # Prepare the input data
    input_data = np.array([[ag, al, b, ba, be, ca, cd, co, cr, cu, fe, h20, k, li, mg, mn, mo, na, ni, p, pb, pqindex, s, sb, si, sn, ti, u100, u75, u50, u25, u20, u14, u6, u4, v, v40, zn]])
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