import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_title="Black-Scholes Option Pricing Model", layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>Black-Scholes Option Price Calculator</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: white;'>Made by Asjad</h6>", unsafe_allow_html=True)
st.write("----")

def black_scholes_calc(S0, K, r, T, sigma, option_type):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    d2 = d1 - sigma*np.sqrt(T)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_d1 = norm.cdf(-d1)
    n_d2 = norm.cdf(-d2)
    c = nd1*S0 - nd2*K*np.exp(-r*T)
    p = K*np.exp(-r*T)*n_d2 - S0*n_d1
    return c if option_type == 'call' else p

def optionGreeks(S0, K, r, T, sigma, option_type="call"):
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r + sigma**2/2)*T)
    d2 = d1 - (sigma*np.sqrt(T))
    
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * np.sqrt(T) * norm.pdf(d1) * 0.01
    theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
             r * K * np.exp(-r*T) * norm.cdf(d2)) / 365 if option_type == "call" else \
            (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
             r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    rho = 0.01 * K * T * np.exp(-r*T) * norm.cdf(d2) if option_type == "call" else \
          0.01 * -K * T * np.exp(-r*T) * norm.cdf(-d2)
    
    return delta, gamma, theta, vega, rho

st.sidebar.header("Input Parameters")
S0 = st.sidebar.slider("Current Stock Price (S)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
K = st.sidebar.slider("Strike Price (K)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
T = st.sidebar.slider("Time to Maturity (T) in years", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
r = st.sidebar.slider("Risk-Free Interest Rate (r)", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
sigma = st.sidebar.slider("Volatility (σ)", min_value=0.1, max_value=1.0, value=0.2, step=0.01)

call_price = black_scholes_calc(S0, K, r, T, sigma, 'call')
put_price = black_scholes_calc(S0, K, r, T, sigma, 'put')
call_greeks = optionGreeks(S0, K, r, T, sigma, "call")
put_greeks = optionGreeks(S0, K, r, T, sigma, "put")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option")
    st.metric("Price", f"${call_price:.2f}", delta=f"{call_price-K:.2f}")
    st.metric("Delta", f"{call_greeks[0]:.4f}")
    st.metric("Gamma", f"{call_greeks[1]:.4f}")
    st.metric("Theta", f"{call_greeks[2]:.4f}")
    st.metric("Vega", f"{call_greeks[3]:.4f}")
    st.metric("Rho", f"{call_greeks[4]:.4f}")

with col2:
    st.subheader("Put Option")
    st.metric("Price", f"${put_price:.2f}", delta=f"{put_price-K:.2f}")
    st.metric("Delta", f"{put_greeks[0]:.4f}")
    st.metric("Gamma", f"{put_greeks[1]:.4f}")
    st.metric("Theta", f"{put_greeks[2]:.4f}")
    st.metric("Vega", f"{put_greeks[3]:.4f}")
    st.metric("Rho", f"{put_greeks[4]:.4f}")

st.write("----")
st.header("Visualizations")

st.subheader("Option Price vs. Stock Price")
stock_prices = np.linspace(max(1, S0 - 50), S0 + 50, 100)
call_prices = [black_scholes_calc(s, K, r, T, sigma, 'call') for s in stock_prices]
put_prices = [black_scholes_calc(s, K, r, T, sigma, 'put') for s in stock_prices]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(stock_prices, call_prices, label='Call Option', color='#FF6B6B')
ax.plot(stock_prices, put_prices, label='Put Option', color='#4ECDC4')
ax.set_xlabel('Stock Price')
ax.set_ylabel('Option Price')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F0F0F0')
st.pyplot(fig)

st.subheader("Sensitivity Analysis")
param_range = np.linspace(0.5, 1.5, 100)

def sensitivity_analysis(param):
    if param == 'S0':
        return [black_scholes_calc(S0 * p, K, r, T, sigma, 'call') for p in param_range]
    elif param == 'K':
        return [black_scholes_calc(S0, K * p, r, T, sigma, 'call') for p in param_range]
    elif param == 'r':
        return [black_scholes_calc(S0, K, max(0.001, r * p), T, sigma, 'call') for p in param_range]
    elif param == 'T':
        return [black_scholes_calc(S0, K, r, max(0.001, T * p), sigma, 'call') for p in param_range]
    elif param == 'sigma':
        return [black_scholes_calc(S0, K, r, T, max(0.001, sigma * p), 'call') for p in param_range]

params = ['S0', 'K', 'r', 'T', 'sigma']
sensitivities = {param: sensitivity_analysis(param) for param in params}

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FDCB6E', '#6C5B7B']
for param, color in zip(params, colors):
    ax.plot(param_range, sensitivities[param], label=param, color=color)
ax.set_xlabel('Parameter Change (%)')
ax.set_ylabel('Call Option Price')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F0F0F0')
st.pyplot(fig)

st.subheader("Interactive Payoff Diagram")
selected_option = st.radio("Select Option Type", ("Call", "Put"))

stock_price_range = np.linspace(max(1, K - 50), K + 50, 100)
if selected_option == "Call":
    payoff = np.maximum(stock_price_range - K, 0) - call_price
    color = '#FF6B6B'
else:
    payoff = np.maximum(K - stock_price_range, 0) - put_price
    color = '#4ECDC4'

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(stock_price_range, payoff, label=f'{selected_option} Payoff', color=color)
ax.axhline(y=0, color='#45B7D1', linestyle='--')
ax.axvline(x=K, color='#FDCB6E', linestyle='--', label='Strike Price')
ax.set_xlabel('Stock Price at Expiration')
ax.set_ylabel('Profit/Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F0F0F0')
st.pyplot(fig)

st.write("----")
st.header("Compare Multiple Options")

num_options = st.number_input("Number of options to compare", min_value=1, max_value=5, value=2)

options_data = []
for i in range(num_options):
    st.subheader(f"Option {i+1}")
    col1, col2, col3 = st.columns(3)
    with col1:
        option_type = st.selectbox(f"Option Type {i+1}", ("Call", "Put"))
        s = st.number_input(f"Stock Price {i+1}", min_value=1.0, value=S0)
    with col2:
        k = st.number_input(f"Strike Price {i+1}", min_value=1.0, value=K)
        t = st.number_input(f"Time to Maturity {i+1}", min_value=0.1, value=T)
    with col3:
        r_i = st.number_input(f"Interest Rate {i+1}", min_value=0.0, max_value=1.0, value=r)
        sig = st.number_input(f"Volatility {i+1}", min_value=0.01, max_value=1.0, value=sigma)
    
    price = black_scholes_calc(s, k, r_i, t, sig, option_type.lower())
    greeks = optionGreeks(s, k, r_i, t, sig, option_type.lower())
    options_data.append({
        "Type": option_type,
        "Price": price,
        "Delta": greeks[0],
        "Gamma": greeks[1],
        "Theta": greeks[2],
        "Vega": greeks[3],
        "Rho": greeks[4]
    })

comparison_df = pd.DataFrame(options_data)
st.write(comparison_df.style.background_gradient(cmap='viridis'))

st.subheader("Option Price Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=comparison_df.index, y="Price", hue="Type", data=comparison_df, ax=ax, palette=['#FF6B6B', '#4ECDC4'])
ax.set_xlabel("Option")
ax.set_ylabel("Price")
ax.set_facecolor('#F0F0F0')
st.pyplot(fig)

