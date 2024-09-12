import pandas as pd
from pygam import LinearGAM, s, f, l
import numpy as np

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

x = data[['month','day','hour']]
y = data[['trips']]

model = LinearGAM(s(0) + s(1) + f(2))
model = model.gridsearch(x.values, y)
