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

# st.title("Data 1 Analysis")

# # 데이터 로드
# data = load_data('data/data1_imputed.csv')

# # 선택된 변수들 추출
# selected_columns = ['AL', 'BA', 'SB', 'CR', 'ZN', 'Y_LABEL']
# data = data[selected_columns]

# # # 데이터프레임 표시
# # st.dataframe(data)

# # 특징과 타겟 정의
# X = data.drop(columns=['Y_LABEL'])
# y = data['Y_LABEL']

# # 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 모델 훈련
# model = XGBClassifier()
# model.fit(X_train, y_train)

# # 예측 및 평가
# y_pred = model.predict(X_test)
# f1 = f1_score(y_test, y_pred)

# st.write(f"F1 점수: {f1}")

# # 트리 플롯
# st.subheader("모델 트리 플롯")
# fig, ax = plt.subplots(figsize=(20, 10))
# plot_tree(model, num_trees=0, ax=ax)
# st.pyplot(fig)

# # 혼동 행렬
# st.subheader("혼동 행렬")
# cm = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
# ax.set_xlabel('예측')
# ax.set_ylabel('실제')
# st.pyplot(fig)

# # 피처 중요도 그래프
# st.subheader("피처 중요도")
# importance = model.feature_importances_
# features = X.columns
# importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# fig, ax = plt.subplots()
# sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
# ax.set_title('Feature Importance')
# st.pyplot(fig)

# # # AL, BA, SB, CR, ZN 값 입력 및 Y_LABEL 예측
# # st.subheader("값을 입력하여 Y_LABEL 예측")
# # al_value = st.number_input('AL 값을 입력하세요', value=float(data['AL'].mean()))
# # ba_value = st.number_input('BA 값을 입력하세요', value=float(data['BA'].mean()))
# # sb_value = st.number_input('SB 값을 입력하세요', value=float(data['SB'].mean()))
# # cr_value = st.number_input('CR 값을 입력하세요', value=float(data['CR'].mean()))
# # zn_value = st.number_input('ZN 값을 입력하세요', value=float(data['ZN'].mean()))

# # input_data = pd.DataFrame({'AL': [al_value], 'BA': [ba_value], 'SB': [sb_value], 'CR': [cr_value], 'ZN': [zn_value]})
# # y_label_pred = model.predict(input_data)[0]
# # st.write(f"입력된 값에 대한 Y_LABEL 예측 결과: {y_label_pred}")

# st.title('건설장비 고장 예측')
# st.divider()

# st.write("""
# 건설 장비의 오일 샘플에서 원소 농도를 입력하면 장비의 고장 가능성을 예측할 수 있습니다.
# 아래에 각 원소의 농도를 입력하세요:
# """)

# # Input fields for the elements
# al_default = data['AL'].mean()
# ba_default = data['BA'].mean()
# sb_default = data['SB'].mean()
# cr_default = data['CR'].mean()
# zn_default = data['ZN'].mean()

# al = st.number_input('알루미늄 (Al)', min_value=0.0, max_value=10000.0, step=0.1, value=al_default)
# ba = st.number_input('바륨 (Ba)', min_value=0.0, max_value=10000.0, step=0.1, value=ba_default)
# sb = st.number_input('안티모니 (Sb)', min_value=0.0, max_value=10000.0, step=0.1, value=sb_default)
# cr = st.number_input('크롬 (Cr)', min_value=0.0, max_value=10000.0, step=0.1, value=cr_default)
# zn = st.number_input('아연 (Zn)', min_value=0.0, max_value=10000.0, step=0.1, value=zn_default)

# if st.button('예측'):
#     # Prepare the input data
#     input_data = np.array([[al, ba, sb, cr, zn]])
#     # Make prediction
#     prediction = model.predict(input_data)

#     if prediction[0] == 1:
#         st.write("🔴 건설 장비에 고장이 발생할 가능성이 높습니다.")
#     else:
#         st.write("🟢 건설 장비가 정상 작동할 가능성이 높습니다.")

st.title("Data 4 Analysis")

# 데이터 로드
data = load_data('data/data4_imputed.csv')

# # 선택된 변수들 추출
# selected_columns = ['AL', 'BA', 'SB', 'CR', 'ZN', 'Y_LABEL']
# data = data[selected_columns]

# # 데이터프레임 표시
# # st.dataframe(data)

# # 특징과 타겟 정의
# X = data.drop(columns=['Y_LABEL'])
# y = data['Y_LABEL']

# # 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 모델 훈련
# model = XGBClassifier()
# model.fit(X_train, y_train)

# # 예측 및 평가
# y_pred = model.predict(X_test)
# f1 = f1_score(y_test, y_pred)

# st.write(f"F1 점수: {f1}")

# # 트리 플롯
# st.subheader("모델 트리 플롯")
# fig, ax = plt.subplots(figsize=(20, 10))
# plot_tree(model, num_trees=0, ax=ax)
# st.pyplot(fig)

# # 혼동 행렬
# st.subheader("혼동 행렬")
# cm = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
# ax.set_xlabel('예측')
# ax.set_ylabel('실제')
# st.pyplot(fig)

# # 피처 중요도 그래프
# st.subheader("피처 중요도")
# importance = model.feature_importances_
# features = X.columns
# importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# fig, ax = plt.subplots()
# sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
# ax.set_title('Feature Importance')
# st.pyplot(fig)

# # UI 섹션
# st.title('건설장비 고장 예측')
# st.divider()

# st.write("""
# 건설 장비의 오일 샘플에서 원소 농도를 입력하면 장비의 고장 가능성을 예측할 수 있습니다.
# 아래에 각 원소의 농도를 입력하세요:
# """)

# # Input fields for the elements
# al_default = data['AL'].mean()
# ba_default = data['BA'].mean()
# sb_default = data['SB'].mean()
# cr_default = data['CR'].mean()
# zn_default = data['ZN'].mean()

# al = st.number_input('알루미늄 (Al)', min_value=0.0, max_value=10000.0, step=0.1, value=al_default)
# ba = st.number_input('바륨 (Ba)', min_value=0.0, max_value=10000.0, step=0.1, value=ba_default)
# sb = st.number_input('안티모니 (Sb)', min_value=0.0, max_value=10000.0, step=0.1, value=sb_default)
# cr = st.number_input('크롬 (Cr)', min_value=0.0, max_value=10000.0, step=0.1, value=cr_default)
# zn = st.number_input('아연 (Zn)', min_value=0.0, max_value=10000.0, step=0.1, value=zn_default)

# if st.button('예측'):
#     # Prepare the input data
#     input_data = np.array([[al, ba, sb, cr, zn]])
#     # Make prediction
#     prediction_proba = model.predict_proba(input_data) * 100
#     prediction = model.predict(input_data)

#     # Display prediction
#     st.write(f"고장이 발생할 확률: {prediction_proba[0][1]:.2f} %")

#     if prediction[0] == 1:
#         st.write("🔴 건설 장비에 고장이 발생할 가능성이 높습니다.")
#     else:
#         st.write("🟢 건설 장비가 정상 작동할 가능성이 높습니다.")


# st.title('유지보수 비용 절감 계산기')
# st.divider()

# st.write("""
# 예측 알고리즘을 통해 유지보수 비용 절감 효과를 계산해보세요.
# """)

# # Input fields for maintenance costs
# maintenance_cost = st.number_input('고장 후 수리 비용 (만원)', min_value=0.0, max_value=10000.0, step=0.1, value=500.0)
# preventive_maintenance_cost = st.number_input('예방 유지보수 비용 (만원)', min_value=0.0, max_value=10000.0, step=0.1, value=100.0)
# failure_rate = st.number_input('고장 확률 (%)', min_value=0.0, max_value=100.0, step=0.1, value=20.0)

# if st.button('비용 절감 계산'):
#     # Calculate cost savings
#     total_maintenance_cost = maintenance_cost * (failure_rate / 100)
#     cost_savings = total_maintenance_cost - preventive_maintenance_cost

#     if cost_savings > 0:
#         st.write(f"💰 예측 알고리즘을 사용하여 {cost_savings:.2f}만원의 비용을 절감할 수 있습니다.")
#     else:
#         st.write(f"🔴 예방 유지보수가 비용 절감에 도움이 되지 않습니다. 유지보수 전략을 재검토하세요.")

# YEAR 선택 및 Y_LABEL이 1인 데이터 출력
st.title('YEAR 선택에 따른 Y_LABEL 데이터')
year_selected = st.selectbox('YEAR를 선택하세요', sorted(data['YEAR'].unique()))

filtered_data = data[(data['YEAR'] == year_selected) & (data['Y_LABEL'] == 1)]
st.write(f"{year_selected}년에 고장이 발생한 데이터:")
st.dataframe(filtered_data)