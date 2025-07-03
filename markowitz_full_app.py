# markowitz_full_app.py

# ============================================
# ✅ Cài thư viện cần thiết (chạy 1 lần duy nhất)
# pip install streamlit vnstock pandas numpy matplotlib plotly scipy
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import minimize
from vnstock import Vnstock

# ===========================
# ⚙️ Thiết lập giao diện Streamlit
# ===========================
st.set_page_config(page_title="Markowitz Portfolio App", layout="wide")
st.title("📈 Markowitz Portfolio Optimization (VN Stocks)")

# ===========================
# 🔽 Nhập danh sách mã cổ phiếu
# ===========================
tickers_input = st.text_input("Nhập mã cổ phiếu (phân cách bằng dấu phẩy):", "FPT,VNM,VCB")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = "2020-01-01"
end_date = "2024-12-31"
trading_days = 252  # số ngày giao dịch mỗi năm

@st.cache_data
def load_close_price(ticker):
    stock = Vnstock().stock(symbol=ticker, source='VCI')
    df = stock.quote.history(start=start_date, end=end_date)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df[['close']].rename(columns={'close': ticker})

# ===========================
# 📉 Tải dữ liệu giá và tính log return
# ===========================
if tickers:
    price_data = pd.DataFrame()
    for ticker in tickers:
        try:
            df = load_close_price(ticker)
            price_data = pd.concat([price_data, df], axis=1)
        except Exception as e:
            st.warning(f"Không thể lấy dữ liệu {ticker}: {e}")

    if not price_data.empty:
        st.subheader("📊 Biểu đồ giá cổ phiếu")
        st.line_chart(price_data)

        # Tính log return
        log_returns = np.log(price_data / price_data.shift(1)).dropna()

        # ===========================
        # 📌 Tính toán thống kê theo năm
        # ===========================
        mean_returns = log_returns.mean() * trading_days
        variances = log_returns.var() * trading_days
        std_devs = log_returns.std() * np.sqrt(trading_days)
        cov_matrix = log_returns.cov() * trading_days
        corr_matrix = log_returns.corr()

        st.subheader("📈 Thống kê từng cổ phiếu (tính theo năm)")
        stats_df = pd.DataFrame({
            "TSSL kỳ vọng (%)": mean_returns * 100,
            "Phương sai": variances,
            "Độ lệch chuẩn (%)": std_devs * 100
        })
        st.dataframe(stats_df.style.format("{:.4f}"))

        st.subheader("🧮 Ma trận hiệp phương sai")
        st.dataframe(cov_matrix.style.format("{:.4f}"))

        st.subheader("🔗 Ma trận hệ số tương quan")
        st.dataframe(corr_matrix.style.format("{:.2f}"))

        # ===========================
        # 💼 Hàm tối ưu hóa danh mục
        # ===========================
        def portfolio_perf(weights):
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return ret, vol

        def efficient_frontier(num_portfolios=100):
            results = []
            weight_list = []
            for _ in range(num_portfolios):
                weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
                ret, vol = portfolio_perf(weights)
                results.append((ret, vol))
                weight_list.append(weights)
            return pd.DataFrame(results, columns=['Return', 'Risk']), weight_list

        ef_df, weight_list = efficient_frontier(3000)

        st.subheader("🌈 Đường biên hiệu quả (Efficient Frontier)")
        fig = px.scatter(ef_df, x='Risk', y='Return',
                         hover_data={tickers[i]: [f"{w[i]*100:.2f}%" for w in weight_list] for i in range(len(tickers))},
                         labels={'Risk': 'Rủi ro (%)', 'Return': 'TSSL kỳ vọng (%)'})
        fig.update_traces(marker=dict(color='blue'))
        st.plotly_chart(fig, use_container_width=True)

        # ===========================
        # 🎯 Hàm hữu dụng nhà đầu tư
        # ===========================
        st.subheader("🧠 Chọn mức độ e ngại rủi ro")
        A = st.slider("Hệ số A (e ngại rủi ro):", 1, 10, 4, help="A thấp → chấp nhận rủi ro cao. A cao → rất ngại rủi ro")

        ef_df['Utility'] = ef_df['Return'] - 0.5 * A * (ef_df['Risk'] ** 2)
        max_idx = ef_df['Utility'].idxmax()
        opt_weights = weight_list[max_idx]
        opt_return, opt_risk = ef_df.loc[max_idx, ['Return', 'Risk']]

        st.subheader("✅ Danh mục tối ưu theo hàm hữu dụng")
        st.dataframe(pd.DataFrame({"Cổ phiếu": tickers, "Trọng số": np.round(opt_weights, 4)}))
        st.markdown(f"**TSSL kỳ vọng:** {opt_return*100:.2f}%")
        st.markdown(f"**Rủi ro:** {opt_risk*100:.2f}%")
        st.markdown(f"**Hữu dụng:** {ef_df.loc[max_idx, 'Utility']:.4f}")

    else:
        st.warning("Không có dữ liệu cổ phiếu nào được tải thành công.")
else:
    st.info("Nhập danh sách mã cổ phiếu để bắt đầu.")