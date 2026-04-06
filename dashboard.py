import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as px_go
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(layout="wide", page_title="금융 위기 대응 자산 배분 대시보드")

import os

# 1. 데이터 로드 및 전처리
@st.cache_data
def load_data():
    # 현재 파일(dashboard.py)의 디렉토리를 기준으로 경로 설정
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(curr_dir, 'data', 'finance_2020_data.csv')
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 일일 수익률 계산
    df['USD_Ret'] = df['USD'].pct_change()
    df['Gold_Ret'] = df['Gold'].pct_change()
    df['SP500_Ret'] = df['S&P500'].pct_change()
    
    # 누적 수익률 계산
    df['USD_Cum'] = (1 + df['USD_Ret']).cumprod() - 1
    df['Gold_Cum'] = (1 + df['Gold_Ret']).cumprod() - 1
    df['SP500_Cum'] = (1 + df['SP500_Ret']).cumprod() - 1
    
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 구간 분석 로직
crisis_events = [
    datetime(2020, 3, 11), # COVID-19
    datetime(2023, 3, 10), # SVB 파산
    datetime(2025, 4, 2)   # 미국 관세 발표
]

def get_segment(date):
    for event in crisis_events:
        if event <= date <= event + timedelta(days=30):
            return "Crisis"
        elif event + timedelta(days=30) < date <= event + timedelta(days=90):
            return "Recovery"
    return "Basic"

df_raw['Segment'] = df_raw['Date'].apply(get_segment)

# 3. 헤더
st.title("🛡️ 금융 위기 대응 자산 배분 대시보드")
st.markdown("---")
st.subheader("📍 현재 시장 상황 요약: 2026-02-27 미국-이란 전쟁 발발 시나리오")
st.info("""
**시나리오 개요**: 중동발 지정학적 리스크 확대로 인해 글로벌 금융 시장의 불확실성이 극도로 높아진 상태입니다. 
과거 팬데믹 및 금융 시스템 위기(SVB) 패턴과 유사한 'Crisis' 국면의 진입으로 간주하며, 이에 따른 자산별 방어력과 반등 수익성을 분석합니다.
""")

# 4. 차트 레이아웃
# Row 1: 누적 수익률
st.markdown("### 📈 Chart 1: 자산별 누적 수익률 (2020 - 2026)")
fig1 = px.line(df_raw, x='Date', y=['Gold_Cum', 'USD_Cum', 'SP500_Cum'],
              labels={'value': '누적 수익률 (%)', 'Date': '날짜'},
              title="자산별 누적 성과 비교",
              color_discrete_map={'Gold_Cum': 'gold', 'USD_Cum': 'blue', 'SP500_Cum': 'red'})

events = {
    "🤒COVID-19": "2020-03-11",
    "🏦SVB 파산": "2023-03-10",
    "🪙미국 관세 발표": "2025-04-02",
    "💥미-이란 전쟁(사나리오)": "2026-02-27"
}

for name, date in events.items():
    fig1.add_vline(x=date, line_width=1.5, line_dash="dash", line_color="gray")
    fig1.add_annotation(x=date, text=name, showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color="gray", size=10))

fig1.update_layout(legend_title="자산구분", hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)
st.caption("그래프 해석: 위기 시 금과 달러는 방어 기제 역할을 하며, S&P 500은 낙폭 이후 가파른 우상향 패턴을 보입니다.")

st.markdown("---")

# Row 2: 상관관계 히트맵
st.markdown("### 🗺️ Chart 2: 구간별 자산 상관관계 패턴 비교")
h_col1, h_col2, h_col3 = st.columns(3)
segments = ["Basic", "Crisis", "Recovery"]
cols = [h_col1, h_col2, h_col3]

for seg, col in zip(segments, cols):
    with col:
        seg_df = df_raw[df_raw['Segment'] == seg][['Gold_Ret', 'USD_Ret', 'SP500_Ret']].dropna()
        if not seg_df.empty:
            corr_matrix = seg_df.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title=f"{seg} 구간 상관관계")
            fig_corr.update_layout(coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# Row 3: MDD 분석
st.markdown("### 📉 Chart 3: 위기 구간 자산별 최대 낙폭(MDD)")
def calculate_mdd(series):
    rolling_max = series.cummax()
    return (series - rolling_max) / rolling_max

df_raw['SP500_DD'] = calculate_mdd(df_raw['S&P500'])
df_raw['Gold_DD'] = calculate_mdd(df_raw['Gold'])
df_raw['USD_DD'] = calculate_mdd(df_raw['USD'])

fig4 = px_go.Figure()
fig4.add_trace(px_go.Scatter(x=df_raw['Date'], y=df_raw['SP500_DD'], fill='tozeroy', name='S&P 500 MDD', line=dict(color='red')))
fig4.add_trace(px_go.Scatter(x=df_raw['Date'], y=df_raw['Gold_DD'], fill='tozeroy', name='Gold MDD', line=dict(color='gold')))
fig4.add_trace(px_go.Scatter(x=df_raw['Date'], y=df_raw['USD_DD'], fill='tozeroy', name='USD MDD', line=dict(color='blue')))

# 주요 이벤트 화살표 및 텍스트 추가
mdd_events = {
    "🤒COVID-19": "2020-03-11",
    "🏦SVB 파산": "2023-03-10",
    "🪙미국 관세 발표": "2025-04-02",
    "💥미-이란 전쟁": "2026-02-27"
}

for name, date in mdd_events.items():
    fig4.add_annotation(
        x=date, y=0,
        text=name,
        showarrow=True,
        arrowhead=2,
        ax=0, ay=-40,
        font=dict(size=11, color="black"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1
    )

fig4.update_layout(
    title="자산별 낙폭 추이 (Drawdown)", 
    yaxis_title="낙폭 비율", 
    hovermode="x unified",
    yaxis=dict(range=[-0.35, 0], tickformat=".0%") # 0~-35% 범위 및 퍼센트 형식
)
st.plotly_chart(fig4, use_container_width=True)


st.markdown("---")

# Row 4: 2026 전쟁 발발 시나리오 대응 포트폴리오 전략 (3대 제안)
st.markdown("### 🥧 Chart 4: 위기 대응 투자 성향별 맞춤 포트폴리오")

p_col1, p_col2, p_col3 = st.columns(3)

# 공통 컬러 맵
color_map = {'Gold': 'gold', 'USD': 'blue', 'S&P 500': 'red'}

with p_col1:
    st.markdown("#### 🟢 1. 안정형 (Safe)")
    w_safe = {'Gold': 0.50, 'USD': 0.40, 'S&P 500': 0.10}
    df_safe = pd.DataFrame(list(w_safe.items()), columns=['Asset', 'Weight'])
    fig_safe = px.pie(df_safe, values='Weight', names='Asset', hole=0.4, title="안정형 비중",
                     color='Asset', color_discrete_map=color_map)
    st.plotly_chart(fig_safe, use_container_width=True)
    st.caption("**가이드**: 원금 방어가 최우선인 전략입니다. 금과 달러 비중을 90%로 설정하여 극한의 위기 상황에서도 자산 가치를 보존합니다.")

with p_col2:
    st.markdown("#### 🟡 2. 최적 추천형 (Optimal)")
    w_opt = {'Gold': 0.45, 'USD': 0.35, 'S&P 500': 0.20}
    df_opt = pd.DataFrame(list(w_opt.items()), columns=['Asset', 'Weight'])
    fig_opt = px.pie(df_opt, values='Weight', names='Asset', hole=0.4, title="최적 추천형 비중",
                    color='Asset', color_discrete_map=color_map)
    st.plotly_chart(fig_opt, use_container_width=True)
    st.caption("**가이드**: 리스크 방어와 반등 기회를 동시에 고려한 밸런스 전략입니다. 과거 위기 패턴 분석을 기반으로 배분되었습니다.")

with p_col3:
    st.markdown("#### 🔴 3. 공격형 (Aggressive)")
    w_agg = {'Gold': 0.30, 'USD': 0.30, 'S&P 500': 0.40}
    df_agg = pd.DataFrame(list(w_agg.items()), columns=['Asset', 'Weight'])
    fig_agg = px.pie(df_agg, values='Weight', names='Asset', hole=0.4, title="공격형 비중",
                    color='Asset', color_discrete_map=color_map)
    st.plotly_chart(fig_agg, use_container_width=True)
    st.caption("**가이드**: 위기 직후의 기술적 반등을 극대화하려는 전략입니다. 위험자산(S&P 500) 비중을 40%까지 높여 적극적으로 대응합니다.")

st.markdown("---")

# Row 5: 4/2 실전 성과 분석
st.markdown("### 🎯 Chart 5: 미국-이란 전쟁 위기 구간 포트폴리오 실적 비교")

# 포트폴리오별 수익률 계산 필드 생성
df_raw['Safe_Ret'] = (df_raw['Gold_Ret'] * 0.50 + df_raw['USD_Ret'] * 0.40 + df_raw['SP500_Ret'] * 0.10)
df_raw['Opt_Ret'] = (df_raw['Gold_Ret'] * 0.45 + df_raw['USD_Ret'] * 0.35 + df_raw['SP500_Ret'] * 0.20)
df_raw['Agg_Ret'] = (df_raw['Gold_Ret'] * 0.30 + df_raw['USD_Ret'] * 0.30 + df_raw['SP500_Ret'] * 0.40)

recent_df = df_raw.tail(30).copy()
target_date = "2026-04-02"
today_data = df_raw[df_raw['Date'] == target_date]

# 6. 2026 미국-이란 전쟁 위기 구간 개별 자산 성과 계산
if not today_data.empty:
    crisis_start_date = "2026-02-27"
    start_data = df_raw[df_raw['Date'] == crisis_start_date]
    
    if not start_data.empty:
        # 시작가 및 종료가 추출
        s_gold, e_gold = start_data['Gold'].values[0], today_data['Gold'].values[0]
        s_sp, e_sp = start_data['S&P500'].values[0], today_data['S&P500'].values[0]
        s_usd, e_usd = start_data['USD'].values[0], today_data['USD'].values[0]
        
        # 변동률 계산
        gold_perf = (e_gold / s_gold) - 1
        sp500_perf = (e_sp / s_sp) - 1
        usd_perf = (e_usd / s_usd) - 1

        st.markdown("#### 🪙 개별 자산 위기 성과")
        a1, a2, a3 = st.columns(3)
        with a1: st.metric("금 (Gold)", f"{(gold_perf*100):.2f}%", f"{gold_perf*100:.2f}%", delta_color="normal")
        with a2: st.metric("S&P 500", f"{(sp500_perf*100):.2f}%", f"{sp500_perf*100:.2f}%", delta_color="normal")
        with a3: st.metric("달러 (USD)", f"{(usd_perf*100):.2f}%", f"{usd_perf*100:.2f}%", delta_color="normal")

    st.markdown("#### 🥧 투자 성향별 포트폴리오 수익률")
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("🛡️ 안정형 (4/2)", f"{(today_data['Safe_Ret'].values[0]*100):.2f}%", "Safe Strategy")
    with m2: st.metric("⭐ 최적 추천형 (4/2)", f"{(today_data['Opt_Ret'].values[0]*100):.2f}%", "Optimal Strategy")
    with m3: st.metric("🔥 공격형 (4/2)", f"{(today_data['Agg_Ret'].values[0]*100):.2f}%", "Aggressive Strategy")

    # 최근 30일 누적 수익률 시뮬레이션
    recent_df['Safe_Cum'] = (1 + recent_df['Safe_Ret']).cumprod() - 1
    recent_df['Opt_Cum'] = (1 + recent_df['Opt_Ret']).cumprod() - 1
    recent_df['Agg_Cum'] = (1 + recent_df['Agg_Ret']).cumprod() - 1
    recent_df['SP500_Cum_Rel'] = (1 + recent_df['SP500_Ret']).cumprod() - 1 # 벤치마크용
    recent_df['Gold_Cum_Rel'] = (1 + recent_df['Gold_Ret']).cumprod() - 1   # 개별 자산 비교용
    recent_df['USD_Cum_Rel'] = (1 + recent_df['USD_Ret']).cumprod() - 1    # 개별 자산 비교용

    fig5 = px.line(recent_df, x='Date', y=['Safe_Cum', 'Opt_Cum', 'Agg_Cum', 'SP500_Cum_Rel', 'Gold_Cum_Rel', 'USD_Cum_Rel'],
                  title="최근 30일간 전략별 및 자산별 누적 성과 비교",
                  color_discrete_map={
                      'Safe_Cum': 'gray', 
                      'Opt_Cum': 'black', 
                      'Agg_Cum': 'purple',
                      'SP500_Cum_Rel': 'red',
                      'Gold_Cum_Rel': 'gold',
                      'USD_Cum_Rel': 'blue'
                  },
                  labels={'value': '누적 수익률 (%)', 'variable': '자산/전략 구분'})
    
    fig5.update_layout(hovermode="x unified")
    st.plotly_chart(fig5, use_container_width=True)
    
    st.info("""
    **분석 결과**: 미국-이란 전쟁 직후 S&P 500(빨간색)의 급락에도 불구하고, **안정형(회색)**과 **최적 추천형(검정색)** 전략은 
    안전자산의 방어력 덕분에 수익률 방어에 우수한 성과를 보입니다. 반면 **공격형(보라색)**은 S&P 500의 영향을 더 많이 받으나, 반등 시 회복 탄력성이 가장 높습니다.
    """)

st.markdown("---")

# Row 6: 회복 국면 추천 포트폴리오 전략 (2단계 제안)
st.markdown("### 🚀 Chart 6: 시장 안정화 단계 추천 포트폴리오 (회복 국면)")

strat_col1, strat_col2 = st.columns(2)

with strat_col1:
    st.markdown("#### 🛡️ 1. 안정 수혜형 (Stable Strategy)")
    stable_weights = {'Gold': 0.35, 'USD': 0.30, 'S&P 500': 0.35}
    stable_df = pd.DataFrame(list(stable_weights.items()), columns=['Asset', 'Weight'])
    fig_stable = px.pie(stable_df, values='Weight', names='Asset', hole=0.4,
                      title="안정 수혜형 비중",
                      color='Asset', color_discrete_map={'Gold': 'gold', 'USD': 'blue', 'S&P 500': 'red'})
    st.plotly_chart(fig_stable, use_container_width=True)
    st.info("**추천 대상**: 원금 보호를 중시하면서 시장 반등의 기회를 놓치고 싶지 않은 보수적 투자자. 자산 간 균형을 통해 변동성을 최소화합니다.")

with strat_col2:
    st.markdown("#### ⚡ 2. 공격 반등형 (Aggressive Strategy)")
    agg_weights = {'Gold': 0.25, 'USD': 0.15, 'S&P 500': 0.60}
    agg_df = pd.DataFrame(list(agg_weights.items()), columns=['Asset', 'Weight'])
    fig_agg = px.pie(agg_df, values='Weight', names='Asset', hole=0.4,
                   title="공격 반등형 비중",
                   color='Asset', color_discrete_map={'Gold': 'gold', 'USD': 'blue', 'S&P 500': 'red'})
    st.plotly_chart(fig_agg, use_container_width=True)
    st.success("**추천 대상**: 위기 이후의 강력한 V자 반등 수익을 극대화하려는 공격적 투자자. S&P 500의 높은 탄력성에 집중 배분합니다.")

st.markdown("---")
st.info(f"📊 데이터 가공 정보: 프로젝트 루트의 finance_2020_data.csv 사용 (총 {len(df_raw)}행)")
