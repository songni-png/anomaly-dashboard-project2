import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="다변량 시계열 이상탐지", layout="wide")

# 1. 가상의 다변량 시계열 샘플 데이터 생성 함수
@st.cache_data
def generate_sample_multivariate_data():
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    # 정상 데이터 생성 (3개의 변수)
    var1 = np.sin(np.linspace(0, 20, 200)) * 10 + np.random.normal(0, 2, 200)
    var2 = np.cos(np.linspace(0, 20, 200)) * 15 + np.random.normal(0, 3, 200)
    var3 = np.linspace(10, 50, 200) + np.random.normal(0, 5, 200)
    
    df = pd.DataFrame({'Date': dates, 'Sensor_A': var1, 'Sensor_B': var2, 'Sensor_C': var3})
    
    # 인위적인 이상치(Anomaly) 주입
    df.loc[50:52, ['Sensor_A', 'Sensor_B']] += 30
    df.loc[150, 'Sensor_C'] -= 40
    return df

st.title("🚨 자동화된 다변량 시계열 이상탐지 대시보드")
st.markdown("임의의 다변량 시계열 CSV 파일을 업로드하면 **Isolation Forest** 알고리즘이 자동으로 이상(Anomaly) 구간을 탐지합니다.")

# 2. 사이드바: 설정 및 파일 업로드
st.sidebar.header("설정 (Settings)")
uploaded_file = st.sidebar.file_uploader("다변량 시계열 CSV 업로드", type=['csv'])

# 이상치 비율(Contamination) 파라미터 조절 슬라이더
st.sidebar.markdown("---")
st.sidebar.subheader("알고리즘 파라미터")
contamination_rate = st.sidebar.slider(
    "예상 이상치 비율 (Contamination)", 
    min_value=0.01, max_value=0.20, value=0.05, step=0.01,
    help="데이터 내에 이상치가 차지하는 대략적인 비율을 설정합니다. 값이 클수록 더 많은 이상치를 탐지합니다."
)

# 3. 데이터 로드
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"'{uploaded_file.name}' 파일이 성공적으로 업로드 및 분석되었습니다!")
else:
    df = generate_sample_multivariate_data()
    st.info("업로드된 파일이 없어 기본 다변량 샘플 데이터를 사용합니다.")

# 데이터 전처리 (첫 번째 열을 날짜/시간 인덱스로 가정, 나머지는 다변량 피처)
time_col = df.columns[0]
feature_cols = df.columns[1:]

st.subheader("데이터 미리보기")
st.write(df.head())

# 4. 모델링 (Isolation Forest를 활용한 다변량 이상탐지)
with st.spinner('다변량 패턴을 분석하여 이상치를 탐지하고 있습니다...'):
    # 피처 데이터 추출
    X = df[feature_cols].copy()
    
    # 결측치 처리 (단순히 0으로 채우거나 앞선 값으로 채움)
    X = X.ffill().fillna(0)
    
    # 모델 정의 및 학습
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    df['Anomaly_Label'] = model.fit_predict(X) # -1: 이상, 1: 정상
    df['Anomaly_Score'] = model.decision_function(X) # 낮을수록 비정상
    
    # 시각화를 위해 이상치 여부를 Boolean으로 변환
    df['Is_Anomaly'] = df['Anomaly_Label'] == -1

# 5. 성능 평가 및 지표 대시보드
st.markdown("---")
st.subheader("📊 이상탐지 결과 요약 및 평가 지표")

total_data = len(df)
anomaly_count = df['Is_Anomaly'].sum()
actual_anomaly_rate = (anomaly_count / total_data) * 100
avg_anomaly_score = df.loc[df['Is_Anomaly'], 'Anomaly_Score'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 데이터 수", f"{total_data} 건")
col2.metric("탐지된 이상치 수", f"{anomaly_count} 건")
col3.metric("실제 탐지 비율", f"{actual_anomaly_rate:.1f} %")
# 이상치 점수가 낮을수록(음수일수록) 심각한 이상치임을 나타냄
col4.metric("평균 이상치 점수", f"{avg_anomaly_score:.2f}")

st.markdown("""
> **💡 탐지 적절성 판단 가이드:**
> * 비지도 학습 특성상 정답(Label)이 없으므로 정밀도나 재현율은 계산할 수 없습니다.
> * 대신 우측의 '이상치 점수 분포'에서 음수 영역(이상치)과 양수 영역(정상)이 뚜렷하게 구분되는지 확인하여 모델의 적절성을 평가합니다.
""")

# 6. 결과 시각화
st.markdown("---")
st.subheader("📈 다변량 시계열 및 이상탐지 시각화")

# 시각화할 피처 선택
selected_feature = st.selectbox("그래프에 표시할 변수(피처)를 선택하세요:", feature_cols)

fig1 = go.Figure()

# 정상 데이터 라인
fig1.add_trace(go.Scatter(
    x=df[time_col], y=df[selected_feature],
    mode='lines', name='정상 데이터',
    line=dict(color='lightblue', width=2)
))

# 이상치 산점도 마커
anomalies = df[df['Is_Anomaly']]
fig1.add_trace(go.Scatter(
    x=anomalies[time_col], y=anomalies[selected_feature],
    mode='markers', name='탐지된 이상치',
    marker=dict(color='red', size=8, symbol='x')
))

fig1.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

# 7. 이상치 점수(Decision Score) 분포도 (모델의 신뢰성 평가용)
st.subheader("📉 이상치 점수(Anomaly Score) 분포")
st.markdown("모델이 계산한 점수입니다. 0보다 작으면 이상치, 0보다 크면 정상으로 분류됩니다.")

fig2 = px.histogram(
    df, x="Anomaly_Score", color="Is_Anomaly", 
    nbins=50, 
    color_discrete_map={True: 'red', False: 'blue'},
    labels={'Is_Anomaly': '이상치 여부', 'Anomaly_Score': '이상치 점수 (낮을수록 이상)'}
)
fig2.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="분류 기준 (0)")
fig2.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig2, use_container_width=True)
