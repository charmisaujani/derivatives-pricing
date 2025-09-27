"""binomial_tree.py

Implementation of European and American option pricing using the Binomial Tree method.

Functions
---------
- binomial_tree_price: main function to price options (European/American)

Usage
-----
Run the module as a script to see example outputs.
"""

import numpy as np
import math

def binomial_tree_price(S, K, T, r, sigma, steps=100, option_type="call", exercise_type="european"):
    """Price an option using the Binomial Tree method.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (continuous)
    sigma : float
        Volatility (annualized)
    steps : int
        Number of time steps
    option_type : str
        'call' or 'put'
    exercise_type : str
        'european' or 'american'

    Returns
    -------
    price : float
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r*dt) - d) / (u - d)

    # initialize asset prices at maturity
    ST = np.array([S * u**j * d**(steps-j) for j in range(steps+1)])

    # initialize option values at maturity
    if option_type.lower() == "call":
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)

    # backward induction
    for i in range(steps-1, -1, -1):
        option_values = np.exp(-r*dt) * (p * option_values[1:i+2] + (1-p) * option_values[0:i+1])
        if exercise_type.lower() == "american":
            # check early exercise
            ST = ST[0:i+1] / u  # step back asset prices
            if option_type.lower() == "call":
                option_values = np.maximum(option_values, ST - K)
            else:
                option_values = np.maximum(option_values, K - ST)

    return float(option_values[0])


if __name__ == "__main__":
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    steps = 100

    euro_call = binomial_tree_price(S, K, T, r, sigma, steps, "call", "european")
    euro_put = binomial_tree_price(S, K, T, r, sigma, steps, "put", "european")
    amer_call = binomial_tree_price(S, K, T, r, sigma, steps, "call", "american")
    amer_put = binomial_tree_price(S, K, T, r, sigma, steps, "put", "american")

    print(f"European Call Price: {euro_call:.4f}")
    print(f"European Put Price : {euro_put:.4f}")
    print(f"American Call Price: {amer_call:.4f}")
    print(f"American Put Price : {amer_put:.4f}")
