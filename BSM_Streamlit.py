import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Black-Scholes Option Pricing Model", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Black-Scholes Option Price Calculator</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: white;'>Made by Asjad</h1>", unsafe_allow_html=True)
st.write("----")

def black_scholes_calc(S0, K, r, T, sigma, option_type):
    '''This function calculates the value of the European option based on Black-Scholes formula'''
    # 1) determine N(d1) and N(d2)
    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r+sigma**2/2)*T)
    d2 = d1 - sigma*np.sqrt(T)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_d1 = norm.cdf(-d1)
    n_d2 = norm.cdf(-d2)
    # 2) determine call value
    c = nd1*S0 - nd2*K*np.exp(-r*T)
    # 3) determine put value
    p = K*np.exp(-r*T)*n_d2 - S0*n_d1
    # 4) define which value to return based on the option_type parameter
    if option_type=='call':
        st.success(c)
    elif option_type=='put':
        st.success(p)

def optionDelta (S0, K, r, T, sigma, option_type="call"):

    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r + sigma**2/2)*T)
    d2 = d1 - (sigma*np.sqrt(T))
    
    if option_type == "call":
        delta = norm.cdf(d1, 0, 1)
    elif option_type == "put":
        delta = -norm.cdf(-d1, 0, 1)

    return delta

def optionGamma (S0, K, r, T, sigma):

    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r + sigma**2/2)*T)
    d2 = d1 - (sigma*np.sqrt(T))

    gamma = norm.pdf(d1, 0, 1)/ (S0* sigma * np.sqrt(T))
    return gamma

def optionTheta(S0, K, r, T, sigma, option_type="c"):

    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r + sigma**2/2)*T)
    d2 = d1 - (sigma*np.sqrt(T))

    if option_type == "call":
        theta = - ((S0 * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2, 0, 1)

    elif option_type == "put":
        theta = - ((S0 * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r*T) * norm.cdf(-d2, 0, 1)
    return theta/365


def optionVega (S0, K, r, T, sigma):

    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r + sigma**2/2)*T)
    d2 = d1 - (sigma*np.sqrt(T))

    vega = S0 * np.sqrt(T) * norm.pdf(d1, 0, 1) * 0.01
    return vega
    

def optionRho(S0, K, r, T, sigma, option_type="call"):

    d1 = 1/(sigma*np.sqrt(T)) * (np.log(S0/K) + (r + sigma**2/2)*T)
    d2 = d1 - (sigma*np.sqrt(T))

    if option_type == "call":
        rho = 0.01 * K * T * np.exp(-r*T) * norm.cdf(d2, 0, 1)
    elif option_type == "put":
        rho = 0.01 * -K * T * np.exp(-r*T) * norm.cdf(-d2, 0, 1)
    return rho



col1, col2 = st.columns(2, gap = 'large')

with col1:
    st.markdown("<h4 style='text-align: center; color: white;'>Enter Input Parameters: </h1>", unsafe_allow_html=True)
    # input parameters
    S0 = st.slider("Current Stock Price (S)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0, format="%.2f")
    K = st.slider("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0, format="%.2f")
    T = st.slider("Time to Maturity (T)", min_value=0.1, max_value=2.0, value=1.0, step=0.1, format="%.2f")
    r = st.slider("Risk-Free Interest Rate (r)", min_value=0.0, max_value=0.2, value=0.06, step=0.005, format="%.2f")
    sigma = st.slider("Volatility (σ)", min_value=0.1, max_value=0.5, value=0.2, step=0.01, format="%.2f")

with col2:
    st.markdown("<h4 style='text-align: center; color: white;'>Calculations: </h1>", unsafe_allow_html=True)

    d1 = 1 / (sigma * np.sqrt(T)) * (np.log(S0 / K) + (r + sigma ** 2 / 2) * T)
    d2 = d1 - (sigma * np.sqrt(T))
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_d1 = norm.cdf(-d1)
    n_d2 = norm.cdf(-d2)

    c = S0 * nd1 - K * nd2 * np.exp(-r * T)
    p = K * np.exp(-r * T) * n_d2 - S0 * n_d1

    c = format(c, ".2f")
    p = format(p, ".2f")

    st.latex(
        r'''d_1 = \frac{1}{\sigma\sqrt{T}} \left(\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T\right)''')
    st.latex(r'''d_2 = d_1 - \sigma\sqrt{T}''')
    st.latex(r'''c = N(d_1)S_0 - N(d_2)Ke^{-rT}''')
    st.latex(r'''p = Ke^{-rT}N(-d_2) - S_0N(-d_1)''')
    st.write("")
    st.write("")
    
    col3, col4, col5, col6, col7 = st.columns(5)
    
    col6.metric(label = "Put Price", value = p)
    col4.metric(label = "Call Price", value = c)

    bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
    bcol1.metric("Delta", str(round(optionDelta(S0, K, r, T, sigma,option_type="call"), 3)))
    bcol2.metric("Gamma", str(round(optionGamma(S0, K, r, T, sigma), 3)))
    bcol3.metric("Theta", str(round(optionTheta(S0, K, r, T, sigma,option_type="call"), 3)))
    bcol4.metric("Vega", str(round(optionVega(S0, K, r, T, sigma), 3)))
    bcol5.metric("Rho", str(round(optionRho(S0, K, r, T, sigma,option_type="call"), 3)))

