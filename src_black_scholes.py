"""black_scholes.py

Implementation of Black-Scholes pricing and Greeks for European options.

Functions
---------
- d1_d2: compute d1 and d2
- bs_price: price a European call/put
- bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho: Greeks

Usage
-----
Run the module as a script to see example outputs.

"""
from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np
from scipy.stats import norm


def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes formula.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity in years (T>0)
    r : float
        Continuously compounded risk-free rate
    sigma : float
        Volatility (annualized)

    Returns
    -------
    (d1, d2)
    """
    if T <= 0:
        raise ValueError("Time to maturity T must be > 0")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be > 0")

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: Union[float, np.ndarray], K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Union[float, np.ndarray]:
    """Price a European option using Black-Scholes closed form.

    Parameters
    ----------
    S : float or np.ndarray
        Spot price (or array of spots)
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (continuous)
    sigma : float
        Volatility (annual)
    option_type : str
        "call" or "put"

    Returns
    -------
    price : float or np.ndarray
    """
    S_arr = np.asarray(S, dtype=float)

    d1, d2 = d1_d2(S_arr if S_arr.size > 1 else float(S_arr), K, T, sigma=sigma, r=r)

    # broadcasting works since norm.cdf accepts arrays
    if option_type.lower() == "call":
        price = S_arr * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S_arr * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(price) if np.isscalar(S) else price


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Delta of a European option."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma of a European option."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega of a European option (per 1.0 vol, not per 1%)."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * math.sqrt(T))


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Theta of a European option (per year)."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    term1 = - (S * pdf_d1 * sigma) / (2 * math.sqrt(T))
    if option_type.lower() == "call":
        term2 = - r * K * math.exp(-r * T) * norm.cdf(d2)
        return float(term1 + term2)
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        return float(term1 + term2)


def bs_rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Rho of a European option (sensitivity to interest rate)."""
    _, d2 = d1_d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return float(K * T * math.exp(-r * T) * norm.cdf(d2))
    else:
        return float(-K * T * math.exp(-r * T) * norm.cdf(-d2))


if __name__ == "__main__":
    # simple demo
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    call = bs_price(S, K, T, r, sigma, option_type="call")
    put = bs_price(S, K, T, r, sigma, option_type="put")

    print(f"European Call Price: {call:.4f}")
    print(f"European Put Price : {put:.4f}")
    print(f"Call Delta: {bs_delta(S,K,T,r,sigma,'call'):.4f}")
    print(f"Call Gamma: {bs_gamma(S,K,T,r,sigma):.6f}")
    print(f"Call Vega : {bs_vega(S,K,T,r,sigma):.4f}")
    print(f"Call Theta: {bs_theta(S,K,T,r,sigma,'call'):.4f}")
    print(f"Call Rho  : {bs_rho(S,K,T,r,sigma,'call'):.4f}")
