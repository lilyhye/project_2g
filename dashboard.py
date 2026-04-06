import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="위기 대응 자산분석 대시보드", layout="wide")

st.title("🛡️ 2020년 이후 자산 데이터 기반 위기 대응 포트폴리오")
st.markdown("""
**📝 시나리오 분석**: 2026년 2월 27일, 미국-이란 간의 지정학적 리스크(가상 전쟁 발발) 상황 가정.
과거 3차례 주요 위기(코로나19, SVB 파산, 미국관세 부과 발표)에서 S&P500, 금, 달러가 보여준 수익 지표와 방어력을 **직접 알고리즘으로 스코어링**하여 현재 적용할 수 있는 최적의 분산 투자 비중 가이드를 제안합니다!
""")

# 1. 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('data/finance_10y_data.csv', parse_dates=['Date'], index_col='Date')
    df.ffill(inplace=True) # 결측치는 전일 종가 기반으로 채움
    return df

try:
    df = load_data()
except Exception as e:
    st.error("데이터 파일('finance_10y_data.csv')을 찾을 수 없습니다.")
    st.stop()

# 일일 수익률 및 누적 수익률 계산
returns = df.pct_change().dropna()
cum_returns = (1 + returns).cumprod() - 1

# 2. 기획안에 따른 시계열 구간 정의 (Segmentation)
crisis_events = {
    'COVID-19': pd.to_datetime('2020-03-11'),
    'SVB Collapse': pd.to_datetime('2023-03-10'),
    'US-Tariff': pd.to_datetime('2025-04-02'),
}

df['Segment'] = 'Basic'
for event, start_date in crisis_events.items():
    crisis_end = start_date + timedelta(days=30)
    recovery_end = crisis_end + timedelta(days=30) # 위기 종료 후 30일을 회복기로 가정
    
    # 해당 날짜 사이를 위기 구간으로 부여
    df.loc[(df.index >= start_date) & (df.index <= crisis_end), 'Segment'] = 'Pandemic/Crisis'
    # 위기가 아니면서 회복기인 날짜 부여
    mask_recovery = (df.index > crisis_end) & (df.index <= recovery_end) & (df['Segment'] != 'Pandemic/Crisis')
    df.loc[mask_recovery, 'Segment'] = 'Recovery'

# ==============================================================
# ✨ 사용자 맞춤형 포트폴리오 추천 알고리즘 로직 ✨
# ==============================================================
# 과거 Crisis로 분류된 날들만의 수익률을 뽑아냄
crisis_mask = df['Segment'] == 'Pandemic/Crisis'
crisis_returns = returns[crisis_mask.reindex(returns.index, fill_value=False)]

# 각 자산의 위기 구간 평균 일일 수익률과 위험도(표준편차)
crisis_mean = crisis_returns.mean()
crisis_std = crisis_returns.std()

# --- [최적 추천형(Balanced)]: 기존 로직 ---
# 샤프 지수 (Sharpe Ratio) = 초과 수익을 위험으로 나눈 값 (여기선 위기 방어력/성능 척도로 사용)
crisis_sharpe = crisis_mean / crisis_std

# Min-Max 기반 점수 조정: 가장 실적이 나빴던 자산을 0으로 맞추고 최소 보정값(0.1) 부여
adjusted_score_balanced = (crisis_sharpe - crisis_sharpe.min()) + 0.1

# 금(Gold)과 달러(Dollar) 같은 전통적 안전자산 프리미엄 부여 (알고리즘적 조정)
safe_assets = ['Gold', 'Dollar']
for asset in safe_assets:
    if asset in adjusted_score_balanced:
        adjusted_score_balanced[asset] *= 1.3 # 방어 목적이므로 안전자산에 30%의 추가 알고리즘 비중 부과

# 최종 비중 산출 (100% 합계 기준)
optimal_weights = (adjusted_score_balanced / adjusted_score_balanced.sum()) * 100

# --- [안정형(Stable)]: 안전자산 프리미엄 극대화 ---
adjusted_score_stable = (crisis_sharpe - crisis_sharpe.min()) + 0.1
for asset in safe_assets:
    if asset in adjusted_score_stable:
        adjusted_score_stable[asset] *= 1.8 # 안전자산에 80% 추가 가중치
optimal_weights_stable = (adjusted_score_stable / adjusted_score_stable.sum()) * 100

# --- [공격형(Aggressive)]: 위험자산(S&P 500) 프리미엄 부여 ---
adjusted_score_aggressive = (crisis_sharpe - crisis_sharpe.min()) + 0.1
if 'S&P500' in adjusted_score_aggressive:
    adjusted_score_aggressive['S&P500'] *= 1.5 # 주식에 50% 추가 가중치
optimal_weights_aggressive = (adjusted_score_aggressive / adjusted_score_aggressive.sum()) * 100
# ==============================================================

st.divider()

# --- Chart 1: 2020년 이후 누적 수익률 비교 ---
st.subheader("📊 1. 3대 주요 자산 2020년 이후 누적 수익률 비교 (Line Chart)")
fig1 = px.line(cum_returns * 100, x=cum_returns.index, y=cum_returns.columns,
               labels={'value':'누적 수익률 (%)', 'Date':'기준 날짜', 'variable':'자산명'},
               title='2020년 이후 자산별 성장 추이 비교')

# 위기 이벤트 수직선 표시 기능 추가
crisis_annotations = {
    'COVID-19 🦠': '2020-03-11',
    'SVB 파산 🏦': '2023-03-10',
    '미국관세부과 ⚔️': '2025-04-02',
    '미·이란 전쟁 🚀': '2026-02-27'
}
for title, date_str in crisis_annotations.items():
    # 날짜 글자를 밀리초(Timestamp) 기반 숫자로 변환하여 글자 위치 연산 에러 방지
    date_timestamp = pd.to_datetime(date_str).timestamp() * 1000
    fig1.add_vline(x=date_timestamp, line_dash="dash", line_color="gray", annotation_text=title, annotation_position="top left")

st.plotly_chart(fig1, use_container_width=True)
st.info("ℹ️ **[데이터 가공 정보]** 총 " + str(len(df)) + " 영업일(Row)의 종가 데이터를 사용하였으며 결측치는 전일 종가(ffill)로 치환했습니다. yfinance에서 가져온 'Close' 열을 바탕으로 시작일 대비 누적 수익률을 연산했습니다. \n\n" 
           "💡 **[그래프 해석 가이드]** 평소 가장 높은 스노우볼 성장을 보이는 주식(S&P500)과, 시장이 출렁일 때 반대로 가치가 튀어 오르는 달러, 그리고 꾸준히 가치를 보존하며 은근한 우상향을 그리는 통화자산(금)의 장기 궤적을 비교해 포트폴리오 다각화의 필요성을 느끼셔야 합니다.")

# --- Chart 2: 구간별 상관관계 히트맵 ---
st.subheader("🔥 2. 시장 국면별 주요 자산의 상관관계 (Heatmap)")
col1, col2, col3 = st.columns(3)
segments = ['Basic', 'Pandemic/Crisis', 'Recovery']
cols = [col1, col2, col3]

for i, seg in enumerate(segments):
    seg_returns = returns[df['Segment'].reindex(returns.index, fill_value=False) == seg]
    corr = seg_returns.corr()
    
    fig2 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    title=f'{seg} 국면의 관계성')
    fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
    cols[i].plotly_chart(fig2, use_container_width=True)

st.info("ℹ️ **[데이터 가공 정보]** 전체 10년치 일별 데이터를 코로나, 전쟁, 은행 파산 등 주요 위기 발생 직후 30일(Crisis), 그 다음 30일(Recovery), 일반장(Basic) 등 3가지 조건으로 쪼갠 후 numpy 상관계수 공식을 대입했습니다.\n\n"
           "💡 **[그래프 해석 가이드]** 붉은색(1에 근접)은 같이 오르고, 푸른색(-1에 근접)은 반대로 움직입니다. 평시(Basic) 지표와 다르게 빨간색 글씨로 변모하는 위기(Crisis) 지표를 살펴보면 S&P 500이 하락할 때 금이나 달러가 역의 상관성(헷지 작용)을 보이며 안전판 역할을 입증함을 나타냅니다.")

# --- Chart 3: 주요 위기 국면 수직선과 최대 낙폭(MDD) 분석 ---
st.subheader("📉 4. 10년 전체 자산별 최대 낙폭 깊이 (MDD Area Chart)")
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / (1 + rolling_max)

fig4 = px.area(drawdown * 100, x=drawdown.index, y=drawdown.columns,
             labels={'value':'낙폭 넓이 (Drawdown %)', 'Date':'시간 흐름', 'variable':'구성 자산'},
             color_discrete_sequence=px.colors.qualitative.Set2)

for title, date_str in crisis_annotations.items():
    date_timestamp = pd.to_datetime(date_str).timestamp() * 1000
    fig4.add_vline(x=date_timestamp, line_dash="dash", line_color="black", annotation_text=title, annotation_position="bottom right")

st.plotly_chart(fig4, use_container_width=True)

st.info("ℹ️ **[데이터 가공 정보]** 주가가 전고점(최고점) 대비 얼마나 떨어졌는지를 계산하는 Drawdown 수식을 전 구간 적용하여, 24~26년 사이 최근 역사까지 빈틈없이 모든 기간의 하락폭을 그렸고 쉽게 참조할 수 있도록 위기(Crisis) 수직선을 병기하였습니다.\n\n"
           "💡 **[그래프 해석 가이드]** 0기준선(천장) 아래로 움푹 파인 면적이 넓고 깊을수록 투자자가 겪은 두려움과 계좌 파괴를 보여줍니다. 주식이 아래로 크게 파열되는 동안 달러와 금은 비교적 0 근처에서 견고하게 면적을 버텨주며, MDD 방어가 포트폴리오 장기 성장의 핵심임을 가시적으로 나타냅니다.")


# --- Chart 4: 최적 분산투자 포트폴리오 제안 ---
st.subheader("🎯 3. 2026 지정학적 위기 강력 대응 '최적 포트폴리오' (Pie Chart)")
fig3 = px.pie(values=optimal_weights.values, names=optimal_weights.index, hole=0.4,
              color_discrete_sequence=px.colors.qualitative.Pastel)
fig3.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig3, use_container_width=True)

st.info("ℹ️ **[데이터 가공 정보]** 과거 3번의 Crisis 구간의 일별 평균 수익률을 변동성으로 나눈 '샤프 지수'를 계산하여 각 자산의 위기 방어 점수를 도출했습니다. 여기에 알고리즘에 기반해 달러/금에 안전자산 프리미엄 30%를 적용한 후 백분율(100%)로 비중 비례 조정하였습니다.\n\n"
           "💡 **[그래프 해석 가이드]** 금번 2026년 가상 전쟁(Crisis) 상황이 닥쳤을 시 제안하는 자산 배분 비중입니다. 단순히 주식 비중을 무작정 줄이는 것이 아니라, 위기 상황에 가장 튼튼히 자라난 안전망에 점수를 할당해 내 계좌 내에서 리스크를 서로 흡수할 수 있는 논리적인 수비 병력을 도출해냈습니다.")

# --- Chart 5: 포트폴리오 백테스팅 (2026-02-27 ~ 2026-04-03) ---
st.subheader("🛡️ 5. 실제 시장 적용 성과 검증 (26.02.27 ~ 26.04.03)")

# 날짜 필터링 (위기 발발일부터 4월 3일까지)
scenario_start = pd.to_datetime('2026-02-27')
scenario_end = pd.to_datetime('2026-04-03')
mask_recent = (returns.index >= scenario_start) & (returns.index <= scenario_end)
recent_returns = returns[mask_recent].copy()

if not recent_returns.empty:
    # 1) 각 뼈대 자산별 누적 수익률
    recent_cum = (1 + recent_returns).cumprod() - 1
    
    # 2) 각 포트폴리오 수익률 계산 (일일수익률 * 추천비중 합산)
    # 2-1) 최적 추천형 (Balanced)
    weights_ratio = optimal_weights / 100
    port_daily_returns = (recent_returns * weights_ratio).sum(axis=1)
    port_cum_returns = (1 + port_daily_returns).cumprod() - 1

    # 2-2) 안정형 (Stable)
    weights_ratio_stable = optimal_weights_stable / 100
    port_daily_returns_stable = (recent_returns * weights_ratio_stable).sum(axis=1)
    port_cum_returns_stable = (1 + port_daily_returns_stable).cumprod() - 1

    # 2-3) 공격형 (Aggressive)
    weights_ratio_aggressive = optimal_weights_aggressive / 100
    port_daily_returns_aggressive = (recent_returns * weights_ratio_aggressive).sum(axis=1)
    port_cum_returns_aggressive = (1 + port_daily_returns_aggressive).cumprod() - 1
    
    # 2-4) 단순 균등 분산(25%씩) 수익률 계산
    equal_daily_returns = recent_returns.mean(axis=1)
    equal_cum_returns = (1 + equal_daily_returns).cumprod() - 1
    
    # 비교를 위해 컬럼 추가
    recent_cum['⚖️ 균등 분배 포트폴리오(25%씩)'] = equal_cum_returns
    recent_cum['🛡️ 안정형 포트폴리오'] = port_cum_returns_stable
    recent_cum['🌟 최적 추천형 포트폴리오'] = port_cum_returns
    recent_cum['🚀 공격형 포트폴리오'] = port_cum_returns_aggressive
    
    # 3) 선 그래프 그리기
    fig5 = px.line(recent_cum * 100, x=recent_cum.index, y=recent_cum.columns,
                 labels={'value':'누적 수익률 (%)', 'Date':'날짜', 'variable':'투자 대상'},
                 title='개별 자산 vs 다양한 포트폴리오 방어력 비교')
    # 포트폴리오 선 강조 (세 가지 포트 강조)
    fig5.update_traces(line=dict(width=4), selector=dict(name='🌟 최적 추천형 포트폴리오'))
    fig5.update_traces(line=dict(width=3, dash='dash'), selector=dict(name='🛡️ 안정형 포트폴리오'))
    fig5.update_traces(line=dict(width=3, dash='dot'), selector=dict(name='🚀 공격형 포트폴리오'))
    st.plotly_chart(fig5, use_container_width=True)
    
    # 4) 4/3일 기준 결과 메트릭 카드 표시
    st.markdown("#### 📌 4월 3일 기준 최종 누적 수익률")
    
    # 표시할 대상 리스트 정의 (개별 자산 3종 + 포트폴리오 3종)
    target_metrics = [
        '🛡️ 안정형 포트폴리오', '🌟 최적 추천형 포트폴리오', '🚀 공격형 포트폴리오',
        'Gold', 'S&P500', 'Dollar'
    ]
    
    # 메트릭 카드 출력 (3개씩 2줄)
    cols = st.columns(3)
    for idx, col_name in enumerate(target_metrics):
        if col_name in recent_cum.columns:
            final_return = recent_cum.iloc[-1][col_name] * 100
            # Gold, S&P500, Dollar는 한글 이름 병기
            display_name = {
                'Gold': '🥇 Gold (금)',
                'S&P500': '📈 S&P 500',
                'Dollar': '💵 Dollar (달러)'
            }.get(col_name, col_name)
            
            cols[idx % 3].metric(label=display_name, value=f"{final_return:.2f}%")
        
    st.info("ℹ️ **[데이터 가공 정보]** 2026년 2월 27일부터 4월 3일까지의 실제 시장 데이터 상에서, 리스크 성향별(안정/최적/공격) 가중치를 반영한 포트폴리오와 주요 개별 자산의 누적 수익률을 동시 산출하였습니다.\n\n"
               "💡 **[그래프 해석 가이드]** '안정형'은 위기 상황에서 방어 기제(금, 달러)를 극대화한 전략이며, '공격형'은 반등 시 시세 차익(S&P 500)을 노리는 전략입니다. 현재 나의 리스크 수용 정도에 따라 4월 3일까지의 성과가 어떻게 달랐는지 비교해 보세요.")
else:
    st.warning("해당 기간의 데이터가 충분하지 않습니다. 파일 내 데이터를 확인해주세요.")

st.divider()

# ==============================================================
# ✨ 회복기(Recovery) 전용 포트폴리오 추천 알고리즘 ✨
# ==============================================================
# "Recovery" 국면으로 묶였던 날짜의 수익률 데이터만 추출
recovery_mask = df['Segment'] == 'Recovery'
recovery_returns = returns[recovery_mask.reindex(returns.index, fill_value=False)]

# 각 자산의 회복 구간 평균 수익률과 변동성(표준편차)
recovery_mean = recovery_returns.mean()
recovery_std = recovery_returns.std()

# 회복 구간 샤프 지수 계산
recovery_sharpe = recovery_mean / recovery_std

# Min-Max 기반 점수 조정 (최소값 0 맞추고 기본 가중치 0.1 부여)
recovery_score = (recovery_sharpe - recovery_sharpe.min()) + 0.1

# 회복기에는 주식장 등 위험자산이 가파르게 반등하므로, 위험자산(S&P 500)에 역으로 프리미엄 부과
risk_assets = ['S&P500']
for asset in risk_assets:
    if asset in recovery_score:
        recovery_score[asset] *= 1.3 # 폭발력을 타기 위해 위험자산에 30%의 추가 알고리즘 비중 부과

# 최종 회복기 비중 산출 (100% 합계 기준)
recovery_optimal_weights = (recovery_score / recovery_score.sum()) * 100
# ==============================================================

# --- Chart 6: 회복기(Recovery) 포트폴리오 제안 및 인사이트 ---
st.subheader("🌅 6. 위기 진정 후 회복(Recovery) 국면 대비 '반등형 포트폴리오' 최적 비중")

col_chart, col_insight = st.columns([1, 1])

with col_chart:
    fig6 = px.pie(values=recovery_optimal_weights.values, names=recovery_optimal_weights.index, hole=0.4,
                  color_discrete_sequence=px.colors.qualitative.Prism,
                  title="강력한 V자 반등을 잡아먹는 회복장 비중")
    fig6.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig6, use_container_width=True)

with col_insight:
    st.markdown("### 💡 [인사이트] 2026 미국-이란 전쟁 회복장 대응 전략")
    st.markdown("""
    과거 위기 직후의 회복장(30일) 수치로 유추해 낸 **투자 행동 가이드라인**입니다.
    
    * **안전자산 수익 기여도 저하**: 위기 최절정기에서 방패가 되어주었던 달러(Dollar)와 같은 현금성 안전자산은 매력을 급격히 상실합니다. 방어용 비중을 신속하게 덜어내야 합니다.
    * **위험자산 V자 급채찍**: 공포심에 가장 심하게 무너졌던 **주식 시장(S&P 500)**이 강한 반발 매수세로 인해 튕겨 오르게 됩니다.
    
    **✅ 최종 요약 (Action Plan)**
    > 미-이란 간의 휴전 협상 소식이 들리는 등 **리스크 완화(Peak-out)** 조짐이 보인다면, 지체 없이 **3번 차트(방어형 헷지)** 비중에서 자금을 빼내 위 **6번 차트(반등형 포트폴리오)**처럼 **S&P 500 
    쪽에 비중을 공격적으로 밀어넣는 스위칭(리밸런싱)** 작업이 반드시 병행되어야 합니다.
    """)
