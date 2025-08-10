"""
ETF Tracker Project
Author: Artthana Ravi
Description:
    - Downloads historical ETF data from Yahoo Finance
    - Calculates daily and cumulative returns
    - Simulates a weighted ETF portfolio
    - Computes risk metrics
    - Visualizes performance
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="ETF Tracker", layout="wide")

st.title(" ETF Tracker Dashboard")

# Sidebar inputs
st.sidebar.header("Portfolio Settings")

etfs = st.sidebar.text_input("Enter ETF tickers (comma separated)", "XLK, SPY, VTI")
etf_list = [e.strip().upper() for e in etfs.split(",")]

period = st.sidebar.selectbox("Select period", ["6mo", "1y", "2y", "5y"])

weights_input = st.sidebar.text_input("Enter weights (comma separated, must sum to 1)", "0.4, 0.4, 0.2")
weights_list = [float(w.strip()) for w in weights_input.split(",")]

# Validate weights
if len(weights_list) != len(etf_list):
    st.error("⚠ Number of weights must match number of ETFs")
    st.stop()

if abs(sum(weights_list) - 1) > 0.001:
    st.error("⚠ Weights must sum to 1. Please adjust.")
    st.stop()

weights_series = pd.Series(weights_list, index=etf_list)

# Download data
data = yf.download(etf_list, period=period)
adj_close = data["Close"]

# Calculate returns
daily_returns = adj_close.pct_change()
cumulative_returns = (1 + daily_returns).cumprod() - 1

# Portfolio returns
portfolio_daily_returns = daily_returns.dot(weights_series)
portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1

# Risk metrics
portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
portfolio_annual_return = portfolio_daily_returns.mean() * 252
sharpe_ratio = portfolio_annual_return / portfolio_volatility
rolling_max = (1 + portfolio_daily_returns).cumprod().cummax()
drawdown = (1 + portfolio_daily_returns).cumprod() / rolling_max - 1
max_drawdown = drawdown.min()

# Display metrics
st.subheader(" Portfolio Risk Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Annualized Return", f"{portfolio_annual_return:.2%}")
col2.metric("Annualized Volatility", f"{portfolio_volatility:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("Max Drawdown", f"{max_drawdown:.2%}")

# Plot ETF cumulative returns
st.subheader("Cumulative Returns of ETFs")
fig, ax = plt.subplots(figsize=(10, 5))
for etf in cumulative_returns.columns:
    ax.plot(cumulative_returns.index, cumulative_returns[etf], label=etf)
ax.set_title("Cumulative Returns")
ax.legend()
st.pyplot(fig)

# Plot portfolio cumulative returns
st.subheader("Portfolio Cumulative Return")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(portfolio_cumulative_returns.index, portfolio_cumulative_returns, label="Portfolio", color="black", linewidth=2)
ax2.set_title("Portfolio Cumulative Return")
ax2.legend()
st.pyplot(fig2)

# Download buttons
st.subheader(" Download Data")
st.download_button("ETF Prices CSV", adj_close.to_csv().encode(), "etf_prices.csv")
st.download_button("Daily Returns CSV", daily_returns.to_csv().encode(), "etf_daily_returns.csv")
st.download_button("Portfolio Daily Returns CSV", portfolio_daily_returns.to_csv().encode(), "portfolio_daily_returns.csv")
