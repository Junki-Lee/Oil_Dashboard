import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib



# st.set_page_config(
#     page_title="LS빅데이터스쿨 2기 3조 대시보드",
#     page_icon="🏭",
# )

# # Streamlit 대시보드
# st.title('건설기계 오일 상태 분류')
# st.balloons()
# st.divider()
# image_url = "https://ifh.cc/g/P1FsJt.jpg"

# # HTML을 사용하여 이미지 크기 조정
# st.markdown(f'<img src="{image_url}" width="700" height="350">', unsafe_allow_html=True)

# st.markdown(

#     """
# ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
# """
# )

# Page configuration
st.set_page_config(
    page_title="LS빅데이터스쿨 2기 3조 대시보드",
    page_icon="🏗️",
)

st.balloons()

# Define the first page (Homepage)

st.title('건설장비 유지보수 시스템')
st.divider()

st.subheader("오일 샘플 분석을 통한 건설장비 고장 예측 시스템")
st.write("""
우리 대시보드에 오신 것을 환영합니다. 
최첨단 오일 샘플 분석 기술을 이용하여 건설 장비의 고장 여부를 실시간으로 예측합니다. 이 시스템은 장비가 최적의 상태를 유지하도록 하여 가동 중단 시간을 줄이고 효율성을 높입니다. 렌탈 회사들이 고객에게 신뢰할 수 있는 장비를 제공할 수 있도록 도와줍니다.
""")

image_url = "https://ifh.cc/g/P1FsJt.jpg"

# HTML을 사용하여 이미지 크기 조정
st.markdown(f'<img src="{image_url}" width="700" height="350">', unsafe_allow_html=True)

st.write("""
### 주요 특징:
- **정밀 오일 분석:** 유압 실린더, 트랙 등 주요 부품의 오일 샘플을 통해 상태를 진단.
- **실시간 모니터링:** 장비 상태를 실시간으로 추적하여 즉각적인 피드백 제공.
- **예측 유지보수:** 데이터를 기반으로 잠재적 문제를 사전에 예측하고 예방.

### 장점:
- **가동 중단 시간 감소:** 장비의 비가동 시간을 최소화하여 운영 효율성 증대.
- **수명 연장:** 정기적인 유지보수와 상태 진단을 통해 장비의 수명 연장.
- **신뢰성 향상:** 장비가 항상 작업에 준비되어 있도록 보장.

우리의 건설장비 유지보수 시스템이 귀사의 비즈니스에 어떻게 도움이 될 수 있는지 알아보려면 오늘 바로 문의하세요.
""")