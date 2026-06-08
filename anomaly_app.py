import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="다변량 시계열 이상탐지", layout="wide")

# 1. 가상의 다변량 시계열 샘플 데이터 생성
@st.cache_data
def generate_sample_multivariate_data():
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    var1 = np.sin(np.linspace(0, 30, 300)) * 10 + np.random.normal(0, 2, 300)
    var2 = np.cos(np.linspace(0, 30, 300)) * 15 + np.random.normal(0, 3, 300)
    var3 = np.linspace(10, 60, 300) + np.random.normal(0, 5, 300)
    
    df = pd.DataFrame({'Date': dates, 'Sensor_A': var1, 'Sensor_B': var2, 'Sensor_C': var3})
    
    # 인위적인 이상치(Anomaly) 주입
    df.loc[70:73, ['Sensor_A', 'Sensor_B']] += 30
    df.loc[220:222, 'Sensor_C'] -= 40
    df.loc[150, ['Sensor_A', 'Sensor_C']] += 50
    return df

st.title("🚨 다변량 시계열 이상탐지 종합 대시보드")
st.markdown("데이터를 업로드하고 알고리즘을 선택하여 다변량 시계열 데이터 내의 이상 패턴을 다각도로 분석하세요.")

# 2. 사이드바: 설정, 파일 업로드 및 알고리즘 선택
st.sidebar.header("⚙️ 분석 설정 (Settings)")
uploaded_file = st.sidebar.file_uploader("다변량 시계열 CSV 업로드", type=['csv'])

st.sidebar.markdown("---")
st.sidebar.subheader("알고리즘 및 파라미터")
algorithm = st.sidebar.selectbox(
    "탐지 알고리즘 선택",
    ["Isolation Forest", "LOF (Local Outlier Factor)"]
)

contamination_rate = st.sidebar.slider(
    "예상 이상치 비율 (Contamination)", 
    min_value=0.01, max_value=0.20, value=0.05, step=0.01
)

# 3. 데이터 로드 및 전처리
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("파일 업로드 완료!")
else:
    df = generate_sample_multivariate_data()
    st.sidebar.info("샘플 데이터 사용 중")

time_col = df.columns[0]
feature_cols = df.columns[1:]

with st.spinner('데이터 전처리 및 이상치 탐지 알고리즘 구동 중...'):
    X = df[feature_cols].copy().ffill().fillna(0)
    
    # 다변량 분석(PCA, LOF 등)을 위한 데이터 스케일링 (중요 포인트)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. 모델링 (선택된 알고리즘 적용)
    if algorithm == "Isolation Forest":
        model = IsolationForest(contamination=contamination_rate, random_state=42)
        df['Anomaly_Label'] = model.fit_predict(X_scaled)
        df['Anomaly_Score'] = model.decision_function(X_scaled)
    else:  # LOF
        model = LocalOutlierFactor(contamination=contamination_rate)
        df['Anomaly_Label'] = model.fit_predict(X_scaled)
        df['Anomaly_Score'] = model.negative_outlier_factor_
    
    df['Is_Anomaly'] = df['Anomaly_Label'] == -1

# 5. 최상단 KPI 대시보드
st.markdown("---")
total_data = len(df)
anomaly_count = df['Is_Anomaly'].sum()
actual_anomaly_rate = (anomaly_count / total_data) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 데이터 포인트", f"{total_data} 건")
col2.metric("탐지된 이상치", f"{anomaly_count} 건")
col3.metric("이상치 비율", f"{actual_anomaly_rate:.1f} %")
col4.metric("적용 알고리즘", algorithm)
st.markdown("<br>", unsafe_allow_html=True)

# 6. 탭(Tabs)을 활용한 입체적 UI 구성
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 탐지 요약 및 히트맵", 
    "📈 변수별 상세 시계열", 
    "🌌 PCA 다변량 시각화", 
    "💾 이상치 보고서 및 다운로드"
])

with tab1:
    st.subheader("탐지 점수 분포 및 변수 상관관계")
    c1, c2 = st.columns(2)
    with c1:
        # 이상치 점수 히스토그램
        fig_hist = px.histogram(
            df, x="Anomaly_Score", color="Is_Anomaly", 
            nbins=50, color_discrete_map={True: 'red', False: 'blue'},
            title="이상치 점수(Anomaly Score) 분포"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        # 다변량 상관관계 히트맵
        corr_matrix = df[feature_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
            title="변수 간 상관관계 분석"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.subheader("변수별 이상치 탐지 시각화")
    selected_feature = st.selectbox("그래프에 표시할 변수를 선택하세요:", feature_cols)
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=df[time_col], y=df[selected_feature], mode='lines', name='정상 데이터', line=dict(color='lightblue', width=2)
    ))
    
    anomalies = df[df['Is_Anomaly']]
    fig_time.add_trace(go.Scatter(
        x=anomalies[time_col], y=anomalies[selected_feature], mode='markers', name='탐지된 이상치', marker=dict(color='red', size=8, symbol='x')
    ))
    fig_time.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig_time, use_container_width=True)

with tab3:
    st.subheader("PCA 기반 다변량 군집 및 이상치 시각화")
    st.markdown("여러 개의 변수를 2차원으로 축소(PCA)하여, 이상치가 일반 데이터 군집에서 얼마나 떨어져 있는지 직관적으로 확인합니다.")
    
    # PCA 수행 (2차원)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    df_pca['Is_Anomaly'] = df['Is_Anomaly']
    df_pca[time_col] = df[time_col]
    
    fig_pca = px.scatter(
        df_pca, x='PCA1', y='PCA2', color='Is_Anomaly', 
        hover_data=[time_col],
        color_discrete_map={True: 'red', False: 'royalblue'},
        title="2D PCA Scatter Plot"
    )
    fig_pca.update_traces(marker=dict(size=8, opacity=0.7))
    st.plotly_chart(fig_pca, use_container_width=True)

with tab4:
    st.subheader("⚠️ Top 10 심각한 이상치 발생 데이터")
    st.markdown("이상치 점수(Anomaly Score)가 가장 낮은(가장 비정상적인) 상위 10개의 데이터를 확인하고 전체 결과를 다운로드할 수 있습니다.")
    
    top_anomalies = df[df['Is_Anomaly']].sort_values("Anomaly_Score", ascending=True).head(10)
    st.dataframe(top_anomalies, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 변경된 부분: CSV 변환 작업을 함수로 묶고 캐싱 처리하여 속도 및 안정성 향상
    @st.cache_data
    def convert_df(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8-sig')

    csv_data = convert_df(df)
    
    st.download_button(
        label="📥 전체 분석 결과 CSV 다운로드",
        data=csv_data,
        file_name='multivariate_anomaly_result.csv',
        mime='text/csv',
    )
