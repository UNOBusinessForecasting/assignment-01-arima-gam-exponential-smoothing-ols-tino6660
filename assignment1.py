import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

model = smf.ols("trips ~ month + day + hour", data=data)
modelFit = model.fit() 

data2 = pd.read_csv("https://github.com/UNOBusinessForecasting/assignment-01-arima-gam-exponential-smoothing-ols-tino6660/blob/main/assignment_data_train.csv")

df = pd.DataFrame(data2, columns=['Timestamp', 'month', 'day', 'hour', 'trips'])

pred = modelFit.predict(data2)
