# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 31.10.2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('C:\\Users\\admin\\Downloads\\archive\\MLTempDataset1.csv')

print("Original Data:")
print(data.head())

# Convert Datetime column
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Set Datetime as index
data = data.set_index('Datetime')

# Extract temperature values
X = data['Hourly_Temp'].dropna()

# Declare required variables and set figure size
N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

# Visualize original data
plt.plot(X)
plt.title('Original Temperature Data')
plt.xlabel('Datetime')
plt.ylabel('Hourly Temperature')
plt.show()

# Plot ACF and PACF for original data
plt.subplot(2, 1, 1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# Use first 1000 observations for ARMA modeling
X_sample = X.iloc[:N]

# Fitting the ARMA(1,1) model
arma11_model = ARIMA(
    X_sample,
    order=(1, 0, 1),
    trend='c'
).fit()

print("\nARMA(1,1) Model Summary:")
print(arma11_model.summary())

# Deriving parameters
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(
    nsample=N
)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1, lags=40)
plt.title('ARMA(1,1) ACF')
plt.show()

plot_pacf(ARMA_1, lags=40)
plt.title('ARMA(1,1) PACF')
plt.show()

# Fitting the ARMA(2,2) model
arma22_model = ARIMA(
    X_sample,
    order=(2, 0, 2),
    trend='c'
).fit()

print("\nARMA(2,2) Model Summary:")
print(arma22_model.summary())

# Deriving parameters
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) Process
ar2 = np.array([
    1,
    -phi1_arma22,
    -phi2_arma22
])

ma2 = np.array([
    1,
    theta1_arma22,
    theta2_arma22
])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(
    nsample=N
)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2, lags=40)
plt.title('ARMA(2,2) ACF')
plt.show()

plot_pacf(ARMA_2, lags=40)
plt.title('ARMA(2,2) PACF')
plt.show()
```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="1270" height="683" alt="image" src="https://github.com/user-attachments/assets/7dd8102d-c4b7-4541-9017-e617ef71c34d" />



Partial Autocorrelation
<img width="1245" height="652" alt="image" src="https://github.com/user-attachments/assets/1623f103-5a9f-4a16-b48c-4268fa0dc2a1" />

Autocorrelation

<img width="1252" height="655" alt="image" src="https://github.com/user-attachments/assets/32518347-e740-4bf4-9cf4-ea68bff62e68" />


SIMULATED ARMA(2,2) PROCESS:
<img width="1267" height="682" alt="image" src="https://github.com/user-attachments/assets/9125e073-927e-4ae4-b292-beb6206bac1e" />

Partial Autocorrelation

<img width="1248" height="661" alt="image" src="https://github.com/user-attachments/assets/6ba71239-d10a-4416-9912-eb4383f44786" />


Autocorrelation
<img width="1253" height="653" alt="image" src="https://github.com/user-attachments/assets/c52250ec-e32b-4d06-aa05-7ee450f493ea" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
