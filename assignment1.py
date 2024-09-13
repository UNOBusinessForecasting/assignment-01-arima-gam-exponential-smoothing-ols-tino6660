import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

model = smf.ols("Trips ~ hour", data=data)
modelFit = model.fit() 

pred = modelFit.predict()
