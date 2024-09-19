import pandas as pd
from pygam import LinearGAM, s, f, l
import numpy as np
import patsy as pt
from statsmodels.tsa.api import SimpleExpSmoothing

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data['Timestamp']= pd.to_numeric(pd.to_datetime(data['Timestamp']))
x = data[['year','month','day','hour']]
y = data['trips']

model = LinearGAM(s(0) + f(1) + f(2) + f(3))
model = model.gridsearch(x.values, y)

modelFit = model.fit(x,y)

data2 = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")

data2['Timestamp']= pd.to_numeric(pd.to_datetime(data2['Timestamp']))
data2['trips'] = data['trips'].astype(float)


pred = modelFit.predict(data2[['year', 'month', 'day', 'hour']])
