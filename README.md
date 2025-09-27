# Derivatives-Pricing

Derivatives Pricing Models in Python -
This project implements fundamental option pricing models in Python, demonstrating core quantitative finance techniques used in trading and risk management. It is designed to showcase both the mathematical foundations and practical implementation of derivatives pricing.

📌 Objectives
Implement and compare different option pricing methods.
Compute sensitivities (Greeks) to understand risk exposures.
Visualize results to build financial intuition.
Provide clean, well-documented code that can be extended for advanced use cases.

📂 Project Structure -

derivatives-pricing/
- notebooks/
  - black_scholes.ipynb       # Closed-form solutions
  - binomial_tree.ipynb       # Lattice method
  - monte_carlo.ipynb         # Simulation approach

- src/
  - black_scholes.py
  - binomial_tree.py
  - monte_carlo.py

- data/                         

- README.md

- requirements.txt

🔧 Models Implemented -

Black–Scholes Formula (European options)
Closed-form pricing for calls and puts.
Calculation of Greeks: Delta, Gamma, Theta, Vega, Rho.

Binomial Tree Method
Flexible lattice approach for American and European options.
Intuitive step-by-step visualization of option value evolution.

Monte Carlo Simulation
Pricing of path-dependent derivatives (e.g., Asian options).
Simulated asset price paths under geometric Brownian motion.

📚 Skills Demonstrated -

Quantitative finance & derivatives pricing
Numerical methods (Monte Carlo, lattice models)
Python for finance (NumPy, SciPy, Matplotlib)
Data visualization for financial insights
Clear documentation and reproducible workflows

🔮 Possible Extensions -

Pricing of exotic options (barrier, lookback, Asian).
Implied volatility surface construction.
Calibration of models to real market data.
Portfolio Greeks aggregation and stress testing.
