import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import plotly.graph_objects as go
import requests
import time

st.set_page_config(page_title="Markowitz Portfolio Optimizer", layout="wide")
st.title("📈 Markowitz Portfolio Optimization - Việt Nam Market (Bypass API)")

# ----- Parameters -----
st.sidebar.header("Thông tin nhập vào")
stock_list = st.sidebar.text_input("Nhập mã cổ phiếu (phân tách bằng dấu phẩy)", "FPT,MWG,VCI,SSI")
start_date = st.sidebar.date_input("Ngày bắt đầu", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Ngày kết thúc", pd.to_datetime("2024-12-31"))

# Risk aversion A
risk_aversion = st.sidebar.slider("Hệ số ưa thích rủi ro (A)", min_value=1, max_value=10, value=4)
st.sidebar.caption("A thấp → thích rủi ro, A cao → rất ngại rủi ro")

# ----- Convert date to timestamp -----
def to_timestamp(date):
    return int(time.mktime(pd.to_datetime(date).timetuple()))

# ----- Custom data fetch from VCI API -----
def load_close_price(symbol):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://online.vndirect.com.vn/"
    }
    url = f"https://services.vndirect.com.vn/chart/api/history?symbol={symbol}&resolution=D&from={to_timestamp(start_date)}&to={to_timestamp(end_date)}"
    try:
        r = requests.get(url, headers=headers)
        data = r.json()
        df = pd.DataFrame(data['data'])
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df = df.set_index('t')
        df = df.rename(columns={'c': symbol})
        return df[[symbol]]
    except Exception as e:
        st.error(f"Không thể tải dữ liệu cho mã {symbol}: {e}")
        return None

# ----- Load data -----
symbols = [sym.strip().upper() for sym in stock_list.split(",") if sym.strip()]
all_data = [load_close_price(sym) for sym in symbols]
all_data = [df for df in all_data if df is not None]

if len(all_data) < 2:
    st.warning("Cần ít nhất 2 mã cổ phiếu để tối ưu danh mục.")
    st.stop()

# ----- Combine & calculate returns -----
data = pd.concat(all_data, axis=1).dropna()
returns = np.log(data / data.shift(1)).dropna()

# Annualized stats
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
std_devs = returns.std() * np.sqrt(252)

# ----- Show basic stats -----
st.subheader("📊 Thống kê từng cổ phiếu")
st.dataframe(pd.DataFrame({
    'Tỷ suất sinh lợi kỳ vọng (E)': mean_returns,
    'Phương sai (σ²)': returns.var() * 252,
    'Độ lệch chuẩn (σ)': std_devs
}))

# ----- Show correlation & covariance -----
st.subheader("📌 Ma trận hệ số tương quan")
st.dataframe(returns.corr())
st.subheader("📌 Ma trận hiệp phương sai")
st.dataframe(cov_matrix)

# ----- Portfolio optimization (SLSQP) -----
def portfolio_perf(weights):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    utility = ret - 0.5 * risk_aversion * vol ** 2
    return -utility

n = len(symbols)
bounds = tuple((0, 1) for _ in range(n))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
init_guess = n * [1. / n, ]
opt_result = minimize(portfolio_perf, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = opt_result.x
opt_ret = np.dot(opt_weights, mean_returns)
opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))

# ----- Efficient frontier -----
def generate_frontier(n_points=100):
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    frontier_vol = []
    frontier_weights = []
    for target in target_returns:
        cons = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
        )
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))), init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            frontier_vol.append(result.fun)
            frontier_weights.append(result.x)
    return target_returns, frontier_vol, frontier_weights

rets, vols, wts = generate_frontier()

# ----- Plot frontier -----
st.subheader("📉 Biên hiệu quả (Efficient Frontier)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=vols, y=rets, mode='lines+markers', name='Efficient Frontier',
                         text=[f"Danh mục: {dict(zip(symbols, np.round(w*100, 2)))}" for w in wts],
                         hoverinfo='text+y'))
fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', name='Danh mục tối ưu',
                         marker=dict(color='red', size=12)))
fig.update_layout(xaxis_title='Rủi ro (Độ lệch chuẩn σ)', yaxis_title='Tỷ suất sinh lợi kỳ vọng', height=600)
st.plotly_chart(fig, use_container_width=True)

# ----- Show optimal weights -----
st.subheader("🎯 Danh mục tối ưu")
st.dataframe(pd.DataFrame({'Cổ phiếu': symbols, 'Tỷ trọng (%)': np.round(opt_weights * 100, 2)}))
