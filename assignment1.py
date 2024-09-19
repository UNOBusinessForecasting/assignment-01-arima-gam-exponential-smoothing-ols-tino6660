import pandas as pd
from pygam import LinearGAM, s, f, l
import numpy as np
from datetime import datetime

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data['Timestamp']= data['Timestamp'].strftime("%d/%m/%Y")
Titles = ['year','month','day','hour']
x = data[Titles]
y = data[['trips']]

model = LinearGAM(s(0) + s(1) + s(2) + s(3))
model = model.gridsearch(x.values, y)

modelFit = model.fit(x,y)

data2 = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data2['Timestamp'] = data2['Timestamp'].strftime("%d/%m/%Y")

data2['Trips'] = ""

pred = modelFit.predict(data2)
