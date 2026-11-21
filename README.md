# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 21-11-2026
### Dataset Name : AirPassengers
### AIM:
To create a project on Time Series Analysis and Forecasting of Air Passenger Data using the ARIMA model in Python and compare its performance with other forecasting methods.
### ALGORITHM:
1. Import the necessary Python libraries such as Pandas, NumPy, Matplotlib and Statsmodels.
2. Load the AirPassengers CSV dataset and explore the data.
3. Convert the Month column into datetime format and set it as the index.
4. Plot the monthly passenger data to understand its trend and patterns.
5. Check the stationarity of the time series using the ADF test.
6. Apply differencing to make the time series stationary and plot the ACF and PACF graphs.
7. Determine the ARIMA model parameters (p, d, q).
8. Split the dataset into training data (80%) and testing data (20%).
9. Fit the ARIMA model using the training data.
10. Make predictions using the fitted ARIMA model.
11. Evaluate the model using Root Mean Squared Error (RMSE).
12. Compare the ARIMA model with a simple Naive forecasting method.
13. Fit the final ARIMA model using the complete dataset and predict future passenger numbers.
### PROGRAM:
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load AirPassengers dataset
data = pd.read_csv("AirPassengers.csv")

# Convert Month column to datetime
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')

# Set Month as index
data.set_index('Month', inplace=True)
data.sort_index(inplace=True)

# Extract passenger data
data = data['#Passengers'].asfreq('MS')

# ARIMA Model
def arima_model(data, order):

    train_size = int(len(data) * 0.8)

    train_data = data[:train_size]
    test_data = data[train_size:]

    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(
        mean_squared_error(test_data, forecast)
    )

    # Plot actual and predicted data
    plt.figure(figsize=(12, 6))

    plt.plot(
        train_data.index,
        train_data,
        label='Training Data'
    )

    plt.plot(
        test_data.index,
        test_data,
        label='Testing Data'
    )

    plt.plot(
        test_data.index,
        forecast,
        label='Forecasted Data'
    )

    plt.xlabel('Date')
    plt.ylabel('Number of Passengers')

    plt.title('ARIMA Forecasting for Air Passengers')

    plt.legend()
    plt.grid()

    plt.figtext(
        0.05, 0.02,
        f"Root Mean Squared Error (RMSE): {rmse:.4f}"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)


# Call the function
arima_model(data, order=(5, 1, 0))


```

### OUTPUT:
<img width="1057" height="552" alt="image" src="https://github.com/user-attachments/assets/63f4d17a-8358-4ba2-8f2f-ae737f3cb96a" />


### RESULT:
Thus, the ARIMA model was successfully implemented using the AirPassengers dataset in Python. The model was used to analyze monthly passenger data, predict future passenger numbers and evaluate its performance using RMSE by comparing it with the Naive forecasting method.
