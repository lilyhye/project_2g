import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as px_go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(layout="wide", page_title="금융 위기 대응 자산 배분 대시보드")

import os

# 1. 데이터 로드 및 전처리
@st.cache_data(ttl=3600) # 1시간 동안 캐시 유지 (실시간 수집 효율화)
def load_data():
    try:
        # 실시간 데이터 수집 티커 설정
        # 금(GC=F), 달러 인덱스(DX-Y.NYB), S&P500(^GSPC)
        tickers = {
            'Gold': 'GC=F',
            'USD': 'DX-Y.NYB',
            'S&P500': '^GSPC'
        }
        
        all_data = pd.DataFrame()
        
        # 야후 파이낸스에서 실시간 데이터 다운로드 (최근 10년)
        for name, ticker in tickers.items():
            ticker_data = yf.download(ticker, period='10y')
            
            if isinstance(ticker_data.columns, pd.MultiIndex):
                close_price = ticker_data['Close'][ticker]
            else:
                close_price = ticker_data['Close']
                
            all_data[name] = close_price
            
        all_data.index.name = 'Date'
        all_data.reset_index(inplace=True)
        df = all_data.ffill()
        
    except Exception as e:
        st.warning(f"실시간 데이터 수집 중 오류가 발생하여 로컬 파일을 로드합니다: {e}")
        # 현재 파일(dashboard.py)의 디렉토리를 기준으로 경로 설정
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(curr_dir, 'data', 'finance_10y_data.csv')
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        # 컬럼명 통일 (Dollar -> USD)
        if 'Dollar' in df.columns:
            df.rename(columns={'Dollar': 'USD'}, inplace=True)
    
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

# 기술적 지표 계산 함수
def calculate_indicators(df, asset_col):
    temp_df = df.copy()
    
    # 이동평균선 (SMA)
    temp_df['SMA50'] = temp_df[asset_col].rolling(window=50).mean()
    temp_df['SMA200'] = temp_df[asset_col].rolling(window=200).mean()
    temp_df['SMA20'] = temp_df[asset_col].rolling(window=20).mean()
    
    # 볼린저 밴드 (20일 기준)
    std = temp_df[asset_col].rolling(window=20).std()
    temp_df['BB_Upper'] = temp_df['SMA20'] + (std * 2)
    temp_df['BB_Lower'] = temp_df['SMA20'] - (std * 2)
    
    # RSI (14일 기준)
    delta = temp_df[asset_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10) # 0으로 나누기 방지
    temp_df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = temp_df[asset_col].ewm(span=12, adjust=False).mean()
    ema26 = temp_df[asset_col].ewm(span=26, adjust=False).mean()
    temp_df['MACD'] = ema12 - ema26
    temp_df['MACD_Signal'] = temp_df['MACD'].ewm(span=9, adjust=False).mean()
    temp_df['MACD_Hist'] = temp_df['MACD'] - temp_df['MACD_Signal']
    
    # 시각화용 시그널 포인트 계산 (전체 기간)
    # 1. 이동평균선 크로스
    temp_df['Prev_SMA50'] = temp_df['SMA50'].shift(1)
    temp_df['Prev_SMA200'] = temp_df['SMA200'].shift(1)
    
    buy_cond = (temp_df['Prev_SMA50'] < temp_df['Prev_SMA200']) & (temp_df['SMA50'] > temp_df['SMA200'])
    sell_cond = (temp_df['Prev_SMA50'] > temp_df['Prev_SMA200']) & (temp_df['SMA50'] < temp_df['SMA200'])
    
    # 2. RSI/BB (간소화를 위해 가격 차트 표시는 이평선 크로스 위주로 하거나 합산 가능)
    # 여기서는 시각화의 명확성을 위해 모든 주요 Buy/Sell 시점을 마커용 컬럼으로 만듭니다.
    temp_df['Signal_Buy'] = np.where(buy_cond | (temp_df['RSI'] < 30) | (temp_df[asset_col] < temp_df['BB_Lower']), temp_df[asset_col], np.nan)
    temp_df['Signal_Sell'] = np.where(sell_cond | (temp_df['RSI'] > 70) | (temp_df[asset_col] > temp_df['BB_Upper']), temp_df[asset_col], np.nan)
    
    return temp_df

# 매매 신호 분석 함수
def get_trading_signals(df, asset_col):
    if len(df) < 201: # SMA 200 계산을 위해 최소 데이터 필요
        return []
        
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    signals = []
    
    # 1. 이동평균선 골든/데드크로스
    if prev_row['SMA50'] < prev_row['SMA200'] and last_row['SMA50'] > last_row['SMA200']:
        signals.append(("🚀 골든크로스 발생", "50일 이평선이 200일 이평선을 상향 돌파했습니다. 장기 상승 추세 전환 신호입니다.", "Buy"))
    elif prev_row['SMA50'] > prev_row['SMA200'] and last_row['SMA50'] < last_row['SMA200']:
        signals.append(("💀 데드크로스 발생", "50일 이평선이 200일 이평선을 하향 돌파했습니다. 장기 하락 추세 전환 신호입니다.", "Sell"))
        
    # 2. RSI 과매수/과매도
    if last_row['RSI'] < 30:
        signals.append(("📉 RSI 과매도", f"RSI가 {last_row['RSI']:.1f}로 30 미만입니다. 기술적 반등 가능성이 높은 구간입니다.", "Buy"))
    elif last_row['RSI'] > 70:
        signals.append(("📈 RSI 과매수", f"RSI가 {last_row['RSI']:.1f}로 70 초과입니다. 단기 조정 가능성이 높은 구간입니다.", "Sell"))
        
    # 3. 볼린저 밴드
    if last_row[asset_col] < last_row['BB_Lower']:
        signals.append(("🔔 BB 하단 돌파", "가격이 볼린저 밴드 하단을 이탈했습니다. 과매도 상태로 반등을 기대할 수 있습니다.", "Buy"))
    elif last_row[asset_col] > last_row['BB_Upper']:
        signals.append(("🔔 BB 상단 돌파", "가격이 볼린저 밴드 상단을 돌파했습니다. 과열 상태로 이익 실현을 고려할 수 있습니다.", "Sell"))
        
    return signals

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
    "COVID-19": "2020-03-11",
    "러-우 전쟁": "2022-02-24",
    "SVB 파산": "2023-03-10",
    "엔 캐리 청산": "2024-08-05",
    "관세 국면": "2025-04-02",
    "미-이란 전쟁": "2026-02-27"
}

for name, date in events.items():
    fig1.add_vline(x=date, line_width=1.5, line_dash="dash", line_color="gray")
    fig1.add_annotation(x=date, y=1, yref="paper", text=name, showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color="gray", size=13))

fig1.update_layout(legend_title="자산구분", hovermode="x unified")
fig1.update_xaxes(range=['2020-01-01', df_raw['Date'].max()])
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
fig4.update_layout(title="자산별 낙폭 추이 (Drawdown)", yaxis_title="낙폭 비율", hovermode="x unified")
fig4.update_xaxes(range=['2020-01-01', df_raw['Date'].max()])

for name, date in events.items():
    fig4.add_vline(x=date, line_width=1.5, line_dash="dash", line_color="gray")
    fig4.add_annotation(x=date, y=1, yref="paper", text=name, showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color="gray", size=13))

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

# Row 5: 실전 성과 분석 (최신 일자 기준)
latest_date_dt = df_raw['Date'].max()
target_date = latest_date_dt.strftime('%Y-%m-%d')
date_display = f"{latest_date_dt.month}/{latest_date_dt.day}"

st.markdown(f"### 🎯 Chart 5: 투자 성향별 포트폴리오 실적 비교 ({target_date} 기준)")

# 포트폴리오별 수익률 계산 필드 생성
df_raw['Safe_Ret'] = (df_raw['Gold_Ret'] * 0.50 + df_raw['USD_Ret'] * 0.40 + df_raw['SP500_Ret'] * 0.10)
df_raw['Opt_Ret'] = (df_raw['Gold_Ret'] * 0.45 + df_raw['USD_Ret'] * 0.35 + df_raw['SP500_Ret'] * 0.20)
df_raw['Agg_Ret'] = (df_raw['Gold_Ret'] * 0.30 + df_raw['USD_Ret'] * 0.30 + df_raw['SP500_Ret'] * 0.40)

recent_df = df_raw.tail(30).copy()
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

    st.markdown(f"#### 🥧 투자 성향별 포트폴리오 수익률 ({date_display})")
    m1, m2, m3 = st.columns(3)
    with m1: st.metric(f"🛡️ 안정형 ({date_display})", f"{(today_data['Safe_Ret'].values[0]*100):.2f}%", "Safe Strategy")
    with m2: st.metric(f"⭐ 최적 추천형 ({date_display})", f"{(today_data['Opt_Ret'].values[0]*100):.2f}%", "Optimal Strategy")
    with m3: st.metric(f"🔥 공격형 ({date_display})", f"{(today_data['Agg_Ret'].values[0]*100):.2f}%", "Aggressive Strategy")

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
    
    # 미-이란 전쟁 발발일 화살표 표기
    fig5.add_vline(x="2026-02-27", line_width=1.5, line_dash="dash", line_color="red")
    fig5.add_annotation(x="2026-02-27", y=1, yref="paper", text="💥미-이란 전쟁", showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color="gray", size=13))
    
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
st.info(f"📊 데이터 정보: 실시간 수집 및 프로젝트 루트의 데이터 사용 (총 {len(df_raw)}행, 기준일: {target_date})")

# Row 7: 전술적 매매 타이밍 분석 (기술적 지표)
st.markdown("### 🔍 Chart 7: 전술적 매매 타이밍 분석 (기술적 지표)")

# 자산 선택
ta_col1, ta_col2 = st.columns([1, 4])
with ta_col1:
    # S&P500, Gold, USD 중 선택 (load_data에서 Dollar를 USD로 변환함)
    selected_asset = st.selectbox("분석 대상 자산 선택", ["S&P500", "Gold", "USD"], index=0)
    
    # 지표 계산
    df_ta = calculate_indicators(df_raw, selected_asset)
    
    # 최근 2주(14일) 범위 내의 매수/매도 시점 추출하여 타이밍 날짜 제시
    two_weeks_ago = df_ta['Date'].max() - pd.Timedelta(days=14)
    recent_signals_df = df_ta[df_ta['Date'] >= two_weeks_ago].copy()
    
    recent_buys = recent_signals_df[recent_signals_df['Signal_Buy'].notna()]
    recent_sells = recent_signals_df[recent_signals_df['Signal_Sell'].notna()]
    
    st.markdown("#### 📅 최근 2주간 매수/매도 타이밍")
    if recent_buys.empty and recent_sells.empty:
        st.info("최근 2주간 특별한 매수/매도 타이밍이 포착되지 않았습니다.")
    else:
        if not recent_buys.empty:
            buy_dates = recent_buys['Date'].dt.strftime('%Y-%m-%d').tolist()
            st.success(f"**추천 매수 일자:** {', '.join(buy_dates)}")
        if not recent_sells.empty:
            sell_dates = recent_sells['Date'].dt.strftime('%Y-%m-%d').tolist()
            st.error(f"**추천 매도 일자:** {', '.join(sell_dates)}")
            
    # 매매 신호 포착
    asset_signals = get_trading_signals(df_ta, selected_asset)
    
    st.markdown("#### 🚩 실시간 매매 신호")
    if not asset_signals:
        st.info("현재 뚜렷한 기술적 신호가 없습니다. (Hold)")
    else:
        for title, desc, action in asset_signals:
            if action == "Buy":
                st.success(f"**{title}**\n\n{desc}")
            else:
                st.error(f"**{title}**\n\n{desc}")

    # 초보자 가이드 추가 (Expander)
    with st.expander("💡 초보자를 위한 기술적 지표 가이드"):
        st.markdown("""
        **1. 볼린저 밴드 (Bollinger Bands)**
        - 가격의 변동 범위를 보여줍니다.
        - **해석**: 가격이 상단 밴드에 닿으면 '과열(매도 검토)', 하단 밴드에 닿으면 '과매도(매수 검토)'로 봅니다.
        
        **2. 이동평균선 (SMA 50/200)**
        - **골든크로스(▲)**: 단기선(50일)이 장기선(200일)을 위로 뚫을 때 -> 강력한 상승 신호
        - **데드크로스(▼)**: 단기선이 장기선을 아래로 뚫을 때 -> 하락 추세 시작
        
        **3. RSI (상대강도지수)**
        - 0~100 사이의 수치로 매수/매도 강도를 나타냅니다.
        - **70 이상**: 과매수 구간 (단기 조정 가능성) -> 매도 신호
        - **30 이하**: 과매도 구간 (기술적 반등 가능성) -> 매수 신호
        
        **4. MACD**
        - 추세의 방향과 강도를 보여주는 지표입니다.
        - 파란색 선(MACD)이 주황색 선(Signal)을 상향 돌파하면 매수 신호로 해석합니다.
        """)

with ta_col2:
    # 최근 1년(252일) 데이터 시각화하여 가독성 확보
    df_plot = df_ta.tail(252).copy()
    
    fig_ta = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.08, 
                          row_heights=[0.5, 0.25, 0.25],
                          subplot_titles=(f"{selected_asset} 가격 및 볼린저 밴드 (최근 1년)", "RSI (14)", "MACD"))
    
    # 1. 가격/BB/이평선
    fig_ta.add_trace(px_go.Scatter(x=df_plot['Date'], y=df_plot[selected_asset], name='Price', line=dict(color='black', width=2)), row=1, col=1)
    fig_ta.add_trace(px_go.Scatter(x=df_plot['Date'], y=df_plot['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
    fig_ta.add_trace(px_go.Scatter(x=df_plot['Date'], y=df_plot['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash', width=1), fill='tonexty'), row=1, col=1)
    fig_ta.add_trace(px_go.Scatter(x=df_plot['Date'], y=df_plot['SMA50'], name='SMA 50', line=dict(color='blue', width=1.5)), row=1, col=1)
    fig_ta.add_trace(px_go.Scatter(x=df_plot['Date'], y=df_plot['SMA200'], name='SMA 200', line=dict(color='red', width=1.5)), row=1, col=1)
    
    # 2. RSI
    fig_ta.add_trace(px_go.Scatter(x=df_plot['Date'], y=df_plot['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig_ta.add_hline(y=70, line=dict(color="red", dash="dash"), row=2, col=1)
    fig_ta.add_hline(y=30, line=dict(color="green", dash="dash"), row=2, col=1)
    
    # 3. MACD
    fig_ta.add_trace(px_go.Scatter(x=df_plot['Date'], y=df_plot['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig_ta.add_trace(px_go.Scatter(x=df_plot['Date'], y=df_plot['MACD_Signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    fig_ta.add_trace(px_go.Bar(x=df_plot['Date'], y=df_plot['MACD_Hist'], name='Histogram', marker_color='gray'), row=3, col=1)
    
    fig_ta.update_layout(height=850, showlegend=True, hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig_ta, use_container_width=True)

st.caption("※ 본 분석은 기술적 지표에 기반한 참고 자료이며, 투자 결정의 최종 책임은 투자자 본인에게 있습니다.")
