import pandas as pd
from pygam import LinearGAM, s, f, l
import numpy as np
from scipy import stats

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data['Timestamp']= pd.to_numeric(pd.to_datetime(data['Timestamp']))
data_edit = data.replace(',', '') 
data1 = float(data_edit) 
x = data1[['year', 'month', 'day', 'hour']]
y = data1['trips']

model = LinearGAM(s(0) + f(1) + f(2) + f(3))
model = model.gridsearch(x.values, y)

modelFit = model.fit(x,y)

data2 = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data2['Timestamp']= pd.to_numeric(pd.to_datetime(data2['Timestamp']))
data2_edit = data.replace(',', '') 
data3 = float(data_edit2) 

data3['Trips'] = ""

pred = modelFit.predict(data3)
