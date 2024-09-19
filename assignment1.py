import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

Titles = ['year','month','day','hour']
x = data[Titles]
y = data[['trips']]

model = LinearGAM(s(0) + s(1) + s(2) + s(3))
model = model.gridsearch(x.values, y)

data2 = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data2['Trips'] = ""

pred = modelFit.predict(data2)
