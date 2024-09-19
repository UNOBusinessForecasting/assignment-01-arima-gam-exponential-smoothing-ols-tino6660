import pandas as pd
from pygam import LinearGAM, s, f, l
import numpy as np

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data['Timestamp']= pd.to_datetime(data['Timestamp'], format = 'mixed')
Titles = ['year','month','day','hour']
x = data[Titles]
y = data[['trips']]

model = LinearGAM(s(0) + s(1) + s(2) + s(3))
model = model.gridsearch(x.values, y)

modelFit = model.fit(x,y)

data2 = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data2['Timestamp']= pd.to_datetime(data2['Timestamp'], format = 'mixed')

data2['Trips'] = ""

pred = modelFit.predict(data2)
