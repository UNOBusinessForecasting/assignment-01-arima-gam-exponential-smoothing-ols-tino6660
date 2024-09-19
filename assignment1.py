import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

model = smf.ols("trips ~ year + month + day + hour", data=data)
modelFit = model.fit() 

data2 = [['1-1-2019 0:00', 1, 1, 0, ], ['1-1-2019 1:00', 1, 1, 1, ], ['1-1-2019 2:00', 1, 1, 2, ], ['1-1-2019 3:00', 1, 1,	3, ], ['1-1-2019 4:00', 1, 1, 4, ], ['1-1-2019 5:00', 1, 1,	5, ],
['1-1-2019 6:00',	1, 1,	6, ], ['1-1-2019 7:00', 1, 1,	7, ], ['1-1-2019 8:00',	1, 1,	8, ], ['1-1-2019 9:00',	1, 1,	9, ], ['1-1-2019 10:00', 1,	1, 10, ], ['1-1-2019 11:00', 1,	1, 11, ],
['1-1-2019 12:00', 1,	1, 12, ], ['1-1-2019 13:00:00', 1, 1,	13, ], ['1-1-2019 14:00:00', 1, 1,	14, ], ['1-1-2019 15:00:00',	1, 1,	15, ], ['1-1-2019 16:00:00', 1, 1, 16],
['1-1-2019 17:00:00', 1, 1, 17], ['1-1-2019  18:00:00',	1, 1,	18, ], ['1-1-2019 19:00:00', 1,	1, 19, ], ['1-1-2019 20:00:00', 1, 1, 20, ], ['1-1-2019 21:00:00', 1, 1, 21, ],
['1-1-2019 22:00:00',	1, 1, 22, ], ['1-1-2019 23:00:00', 1, 1, 23, ], ['1-2-2019 0:00', 1, 2, 0, ], ['1-2-2019 1:00', 1, 1, 1, ], ['1-2-2019 2:00', 1, 2, 2, ], ['1-2-2019 3:00', 1, 2,	3, ], ['1-2-2019 4:00', 1, 2, 4, ], ['1-2-2019 5:00', 1, 2,	5, ],
['1-2-2019 6:00',	1, 2,	6, ], ['1-2-2019 7:00', 1, 2,	7, ], ['1-2-2019 8:00',	1, 2,	8, ], ['1-2-2019 9:00',	1, 2,	9, ], ['1-2-2019 10:00', 1,	2, 10, ], ['1-2-2019 11:00', 1,	2, 11, ],
['1-2-2019 12:00', 1,	2, 12, ], ['1-2-2019 13:00:00', 1, 2,	13, ], ['1-2-2019 14:00:00', 1, 2,	14, ], ['1-2-2019 15:00:00',	1, 2,	15, ], ['1-2-2019 16:00:00', 1, 2, 16],
['1-2-2019 17:00:00', 1, 2, 17], ['1-2-2019  18:00:00',	1, 2,	18, ], ['1-2-2019 19:00:00', 1,	2, 19, ], ['1-2-2019 20:00:00', 1, 2, 20, ], ['1-2-2019 21:00:00', 1, 2, 21, ],
['1-2-2019 22:00:00',	1, 2, 22, ], ['1-2-2019 23:00:00', 1, 2, 23, ], ['1-3-2019 0:00', 1, 3, 0, ], ['1-3-2019 1:00', 1, 3, 1, ], ['1-3-2019 2:00', 1, 3, 2, ], ['1-3-2019 3:00', 1, 3,	3, ], ['1-3-2019 4:00', 1, 3, 4, ], ['1-3-2019 5:00', 1, 3,	5, ],
['1-3-2019 6:00',	1, 3,	6, ], ['1-3-2019 7:00', 1, 3,	7, ], ['1-3-2019 8:00',	1, 3,	8, ], ['1-3-2019 9:00',	1, 3,	9, ], ['1-3-2019 10:00', 1,	3, 10, ], ['1-3-2019 11:00', 1,	3, 11, ],
['1-3-2019 12:00', 1,	3, 12, ], ['1-3-2019 13:00:00', 1, 3,	13, ], ['1-3-2019 14:00:00', 1, 3, 14, ], ['1-3-2019  15:00:00', 1, 3, 15, ], ['1-3-2019 16:00:00', 1, 3, 16],
['1-3-2019 17:00:00', 1, 3, 17], ['1-3-2019  18:00:00',	1, 3,	18, ], ['1-3-2019 19:00:00', 1,	3, 19, ], ['1-3-2019 20:00:00', 1, 3, 20, ], ['1-3-2019 21:00:00', 1, 3, 21, ],
['1-3-2019 22:00:00',	1, 3, 22, ], ['1-3-2019 23:00:00', 1, 3, 23, ], ['1-4-2019 0:00', 1, 4, 0, ], ['1-4-2019 1:00', 1, 4, 1, ], ['1-4-2019 2:00', 1, 4, 2, ], 
['1-4-2019 3:00', 1, 4,	3, ], ['1-4-2019 4:00', 1, 4, 4, ], ['1-4-2019 5:00', 1, 4,	5, ], ['1-4-2019 6:00',	1, 4,	6, ], ['1-4-2019 7:00', 1, 4,	7, ], ['1-4-2019 8:00',	1, 4,	8, ],
['1-4-2019 9:00',	1, 4,	9, ], ['1-4-2019 10:00', 1,	4, 10, ], ['1-4-2019 11:00', 1,	4, 11, ], ['1-4-2019 12:00', 1,	4, 12, ], ['1-4-2019 13:00:00', 1, 4,	13, ], 
['1-4-2019 14:00:00', 1, 4,	14, ], ['1-4-2019 15:00:00',	1, 4,	15, ], ['1-4-2019 16:00:00', 1, 4, 16], ['1-4-2019 17:00:00', 1, 4, 17], ['1-4-2019 18:00:00', 1, 4, 18, ],
['1-4-2019 19:00:00', 1, 4, 19, ], ['1-4-2019 20:00:00', 1, 4, 20, ], ['1-4-2019 21:00:00', 1, 4, 21, ], ['1-4-2019 22:00:00',	1, 4, 22, ], ['1-4-2019 23:00:00', 1, 4, 23, ], 
['1-5-2019 0:00', 1, 5, 0, ], ['1-5-2019 1:00', 1, 5, 1, ], ['1-5-2019 2:00', 1, 5, 2, ], ['1-5-2019 3:00', 1, 5,	3, ], ['1-5-2019 4:00', 1, 5, 4, ], ['1-5-2019 5:00', 1, 5,	5, ], 
['1-5-2019 6:00',	1, 5,	6, ], ['1-5-2019 7:00', 1, 5,	7, ], ['1-5-2019 8:00',	1, 5,	8, ], ['1-5-2019 9:00',	1, 5,	9, ], ['1-5-2019 10:00', 1,	5, 10, ], ['1-5-2019 11:00', 1,	5, 11, ], 
['1-5-2019 12:00', 1,	5, 12, ], ['1-5-2019 13:00:00', 1, 5,	13, ], ['1-5-2019 14:00:00', 1, 5, 14, ], ['1-5-2019 15:00:00',	1, 5,	15, ], ['1-5-2019 16:00:00', 1, 5, 16], 
['1-5-2019 17:00:00', 1, 5, 17], ['1-5-2019 18:00:00', 1, 5, 18, ], ['1-5-2019 19:00:00', 1, 5, 19, ], ['1-5-2019 20:00:00', 1, 5, 20, ], ['1-5-2019 21:00:00', 1, 5, 21, ],
['1-5-2019 22:00:00',	1, 5, 22, ], ['1-5-2019 23:00:00', 1, 5, 23, ], ['1-6-2019 0:00', 1, 6, 0, ], ['1-6-2019 1:00', 1, 6, 1, ], ['1-6-2019 2:00', 1, 6, 2, ],
['1-6-2019 3:00', 1, 6,	3, ], ['1-6-2019 4:00', 1, 6, 4, ], ['1-6-2019 5:00', 1, 6,	5, ], ['1-6-2019 6:00',	1, 6,	6, ], ['1-6-2019 7:00', 1, 6,	7, ], ['1-6-2019 8:00',	1, 6,	8, ], 
['1-6-2019 9:00',	1, 6,	9, ], ['1-6-2019 10:00', 1,	6, 10, ], ['1-6-2019 11:00', 1,	6, 11, ], ['1-6-2019 12:00', 1,	6, 12, ], ['1-6-2019 13:00:00', 1, 6,	13, ],
['1-6-2019 14:00:00', 1, 6,	14, ], ['1-6-2019 15:00:00', 1, 6, 15, ], ['1-6-2019 16:00:00', 1, 6, 16], ['1-6-2019 17:00:00', 1, 6, 17], ['1-6-2019 18:00:00', 1, 6, 18, ],
['1-6-2019 19:00:00', 1, 6, 19, ], ['1-6-2019 20:00:00', 1, 6, 20, ], ['1-6-2019 21:00:00', 1, 6, 21, ], ['1-6-2019 22:00:00',	1, 6, 22, ], ['1-6-2019 23:00:00', 1, 6, 23, ],
['1-7-2019 0:00', 1, 7, 0, ], ['1-7-2019 1:00', 1, 7, 1, ], ['1-7-2019 2:00', 1, 7, 2, ], ['1-7-2019 3:00', 1, 7,	3, ], ['1-7-2019 4:00', 1, 7, 4, ], ['1-7-2019 5:00', 1, 7,	5, ], 
['1-7-2019 6:00',	1, 7,	6, ], ['1-7-2019 7:00', 1, 7,	7, ], ['1-7-2019 8:00',	1, 7,	8, ], ['1-7-2019 9:00',	1, 7,	9, ], ['1-7-2019 10:00', 1,	7, 10, ], ['1-7-2019 11:00', 1,	7, 11, ], 
['1-7-2019 12:00', 1,	7, 12, ], ['1-7-2019 13:00:00', 1, 7,	13, ], ['1-7-2019 14:00:00', 1, 7, 14, ], ['1-7-2019 15:00:00',	1, 7,	15, ], ['1-7-2019 16:00:00', 1, 7, 16], 
['1-7-2019 17:00:00', 1, 7, 17], ['1-7-2019 18:00:00', 1, 7, 18, ], ['1-7-2019 19:00:00', 1, 7, 19, ], ['1-7-2019 20:00:00', 1, 7, 20, ], ['1-7-2019 21:00:00', 1, 7, 21, ],
['1-7-2019 22:00:00',	1, 7, 22, ], ['1-7-2019 23:00:00', 1, 7, 23, ], ['1-8-2019 0:00', 1, 8, 0, ], ['1-8-2019 1:00', 1, 8, 1, ], ['1-8-2019 2:00', 1, 8, 2, ],
['1-8-2019 3:00', 1, 8,	3, ], ['1-8-2019 4:00', 1, 8, 4, ], ['1-8-2019 5:00', 1, 8,	5, ], ['1-8-2019 6:00',	1, 8,	6, ], ['1-8-2019 7:00', 1, 6,	7, ], ['1-8-2019 8:00',	1, 8,	8, ], 
['1-8-2019 9:00',	1, 8,	9, ], ['1-8-2019 10:00', 1,	8, 10, ], ['1-8-2019 11:00', 1,	8, 11, ], ['1-8-2019 12:00', 1,	8, 12, ], ['1-8-2019 13:00:00', 1, 8,	13, ],
['1-8-2019 14:00:00', 1, 8,	14, ], ['1-8-2019 15:00:00', 1, 8, 15, ], ['1-8-2019 16:00:00', 1, 8, 16], ['1-8-2019 17:00:00', 1, 8, 17], ['1-8-2019 18:00:00', 1, 8, 18, ],
['1-8-2019 19:00:00', 1, 8, 19, ], ['1-8-2019 20:00:00', 1, 8, 20, ], ['1-8-2019 21:00:00', 1, 8, 21, ], ['1-8-2019 22:00:00',	1, 8, 22, ], ['1-8-2019 23:00:00', 1, 8, 23, ],
['1-9-2019 0:00', 1, 9, 0, ], ['1-9-2019 1:00', 1, 9, 1, ], ['1-9-2019 2:00', 1, 9, 2, ], ['1-9-2019 3:00', 1, 9,	3, ], ['1-9-2019 4:00', 1, 9, 4, ], ['1-9-2019 5:00', 1, 9,	5, ], 
['1-9-2019 6:00',	1, 9,	6, ], ['1-9-2019 7:00', 1, 9,	7, ], ['1-9-2019 8:00',	1, 9,	8, ], ['1-9-2019 9:00',	1, 9,	9, ], ['1-9-2019 10:00', 1,	9, 10, ], ['1-9-2019 11:00', 1,	9, 11, ], 
['1-9-2019 12:00', 1,	9, 12, ], ['1-9-2019 13:00:00', 1, 9,	13, ], ['1-9-2019 14:00:00', 1, 9, 14, ], ['1-9-2019 15:00:00',	1, 9,	15, ], ['1-9-2019 16:00:00', 1, 9, 16], 
['1-9-2019 17:00:00', 1, 9, 17], ['1-9-2019 18:00:00', 1, 9, 18, ], ['1-9-2019 19:00:00', 1, 9, 19, ], ['1-9-2019 20:00:00', 1, 9, 20, ], ['1-9-2019 21:00:00', 1, 9, 21, ],
['1-9-2019 22:00:00',	1, 9, 22, ], ['1-9-2019 23:00:00', 1, 9, 23, ], ['1-10-2019 0:00', 1, 10, 0, ], ['1-10-2019 1:00', 1, 10, 1, ], ['1-10-2019 2:00', 1, 10, 2, ],
['1-10-2019 3:00', 1, 10,	3, ], ['1-10-2019 4:00', 1, 10, 4, ], ['1-10-2019 5:00', 1, 10,	5, ], ['1-10-2019 6:00', 1, 10,	6, ], ['1-10-2019 7:00', 1, 10, 7, ], ['1-10-2019 8:00',	1, 10,	8, ], 
['1-10-2019 9:00',	1, 10,	9, ], ['1-10-2019 10:00', 1, 10, 10, ], ['1-10-2019 11:00', 1, 10, 11, ], ['1-10-2019 12:00', 1,	10, 12, ], ['1-10-2019 13:00:00', 1, 10,	13, ],
['1-10-2019 14:00:00', 1, 10,	14, ], ['1-10-2019 15:00:00', 1,10, 15, ], ['1-10-2019 16:00:00', 1, 10, 16], ['1-10-2019 17:00:00', 1, 10, 17], ['1-10-2019 18:00:00', 1, 10, 10, ],
['1-10-2019 19:00:00', 1, 10, 19, ], ['1-10-2019 20:00:00', 1, 10, 20, ], ['1-10-2019 21:00:00', 1, 10, 21, ], ['1-10-2019 22:00:00',	1, 10, 22, ], ['1-10-2019 23:00:00', 1, 10, 23, ],
['1-11-2019 0:00', 1, 11, 0, ], ['1-11-2019 1:00', 1, 11, 1, ], ['1-11-2019 2:00', 1, 11, 2, ], ['1-11-2019 3:00', 1, 11,	3, ], ['1-11-2019 4:00', 1, 11, 4, ], ['1-11-2019 5:00', 1, 11,	5, ], 
['1-11-2019 6:00', 1, 11,	6, ], ['1-11-2019 7:00', 1, 11,	7, ], ['1-11-2019 8:00',	1, 11,	8, ], ['1-11-2019 9:00',	1, 11, 9, ], ['1-11-2019 10:00', 1,	11, 10, ], ['1-11-2019 11:00', 1,	11, 11, ], 
['1-11-2019 12:00', 1,	11, 12, ], ['1-11-2019 13:00:00', 1, 11,	13, ], ['1-11-2019 14:00:00', 1, 11, 14, ], ['1-11-2019 15:00:00',	1, 11,	15, ], ['1-11-2019 16:00:00', 1, 11, 16], 
['1-11-2019 17:00:00', 1, 11, 17], ['1-11-2019 18:00:00', 1, 11, 18, ], ['1-11-2019 19:00:00', 1, 11, 19, ], ['1-11-2019 20:00:00', 1, 11, 20, ], ['1-11-2019 21:00:00', 1, 11, 21, ],
['1-11-2019 22:00:00',	1, 11, 22, ], ['1-11-2019 23:00:00', 1, 11, 23, ], ['1-12-2019 0:00', 1, 12, 0, ], ['1-12-2019 1:00', 1, 12, 1, ], ['1-12-2019 2:00', 1, 12, 2, ],
['1-12-2019 3:00', 1, 12,	3, ], ['1-12-2019 4:00', 1, 12, 4, ], ['1-12-2019 5:00', 1, 12,	5, ], ['1-12-2019 6:00', 1, 12,	6, ], ['1-12-2019 7:00', 1, 12, 7, ], ['1-12-2019 8:00',	1, 12,	8, ], 
['1-12-2019 9:00',	1, 12,	9, ], ['1-12-2019 10:00', 1, 12, 10, ], ['1-12-2019 11:00', 1, 12, 11, ], ['1-12-2019 12:00', 1,	12, 12, ], ['1-12-2019 13:00:00', 1, 12, 13, ],
['1-12-2019 14:00:00', 1, 12,	14, ], ['1-12-2019 15:00:00', 1, 12, 15, ], ['1-12-2019 16:00:00', 1, 12, 16], ['1-12-2019 17:00:00', 1, 12, 17], ['1-12-2019 18:00:00', 1, 12, 10, ],
['1-12-2019 19:00:00', 1, 12, 19, ], ['1-12-2019 20:00:00', 1, 12, 20, ], ['1-12-2019 21:00:00', 1, 12, 21, ], ['1-12-2019 22:00:00',	1, 12, 22, ], ['1-12-2019 23:00:00', 1, 12, 23, ],
['1-13-2019 0:00', 1, 13, 0, ], ['1-13-2019 1:00', 1, 13, 1, ], ['1-13-2019 2:00', 1, 13, 2, ], ['1-13-2019 3:00', 1, 13,	3, ], ['1-13-2019 4:00', 1, 13, 4, ], ['1-13-2019 5:00', 1, 13,	5, ], 
['1-13-2019 6:00', 1, 13,	6, ], ['1-13-2019 7:00', 1, 13,	7, ], ['1-13-2019 8:00',	1, 13,	8, ], ['1-13-2019 9:00',	1, 13, 9, ], ['1-13-2019 10:00', 1,	13, 10, ], ['1-13-2019 11:00', 1,	13, 11, ], 
['1-13-2019 12:00', 1, 13, 12, ], ['1-13-2019 13:00:00', 1, 13,	13, ], ['1-13-2019 14:00:00', 1, 13, 14, ], ['1-13-2019 15:00:00',	1, 13, 15, ], ['1-13-2019 16:00:00', 1, 13, 16], 
['1-13-2019 17:00:00', 1, 13, 17], ['1-13-2019 18:00:00', 1, 13, 18, ], ['1-13-2019 19:00:00', 1, 13, 19, ], ['1-13-2019 20:00:00', 1, 13, 20, ], ['1-13-2019 21:00:00', 1, 13, 21, ],
['1-13-2019 22:00:00', 1, 13, 22, ], ['1-13-2019 23:00:00', 1, 13, 23, ], ['1-14-2019 0:00', 1, 14, 0, ], ['1-14-2019 1:00', 1, 14, 1, ], ['1-14-2019 2:00', 1, 14, 2, ],
['1-14-2019 3:00', 1, 14,	3, ], ['1-14-2019 4:00', 1, 14, 4, ], ['1-14-2019 5:00', 1, 14,	5, ], ['1-14-2019 6:00', 1, 14,	6, ], ['1-14-2019 7:00', 1, 14, 7, ], ['1-14-2019 8:00',	1, 14,	8, ], 
['1-14-2019 9:00', 1, 14,	9, ], ['1-14-2019 10:00', 1, 14, 10, ], ['1-14-2019 11:00', 1, 14, 11, ], ['1-14-2019 12:00', 1, 14, 12, ], ['1-14-2019 13:00:00', 1, 14, 13, ],
['1-14-2019 14:00:00', 1, 14,	14, ], ['1-14-2019 15:00:00', 1, 14, 15, ], ['1-14-2019 16:00:00', 1, 14, 16], ['1-14-2019 17:00:00', 1, 14, 17], ['1-14-2019 18:00:00', 1, 14, 10, ],
['1-14-2019 19:00:00', 1, 14, 19, ], ['1-14-2019 20:00:00', 1, 14, 20, ], ['1-14-2019 21:00:00', 1, 14, 21, ], ['1-14-2019 22:00:00',	1, 14, 22, ], ['1-14-2019 23:00:00', 1, 14, 23, ],
['1-15-2019 0:00', 1, 15, 0, ], ['1-15-2019 1:00', 1, 15, 1, ], ['1-15-2019 2:00', 1, 15, 2, ], ['1-15-2019 3:00', 1, 15,	3, ], ['1-15-2019 4:00', 1, 15, 4, ], ['1-15-2019 5:00', 1, 15,	5, ], 
['1-15-2019 6:00', 1, 15,	6, ], ['1-15-2019 7:00', 1, 15,	7, ], ['1-15-2019 8:00',	1, 15, 8, ], ['1-15-2019 9:00',	1, 15, 9, ], ['1-15-2019 10:00', 1,	15, 10, ], ['1-15-2019 11:00', 1,	15, 11, ], 
['1-15-2019 12:00', 1, 15, 12, ], ['1-15-2019 13:00:00', 1, 15,	13, ], ['1-15-2019 14:00:00', 1, 15, 14, ], ['1-15-2019 15:00:00',	1, 15, 15, ], ['1-15-2019 16:00:00', 1, 15, 16], 
['1-15-2019 17:00:00', 1, 15, 17], ['1-15-2019 18:00:00', 1, 15, 18, ], ['1-15-2019 19:00:00', 1, 15, 19, ], ['1-15-2019 20:00:00', 1, 15, 20, ], ['1-15-2019 21:00:00', 1, 15, 21, ],
['1-15-2019 22:00:00', 1, 15, 22, ], ['1-15-2019 23:00:00', 1, 15, 23, ], ['1-16-2019 0:00', 1, 16, 0, ], ['1-16-2019 1:00', 1, 16, 1, ], ['1-16-2019 2:00', 1, 16, 2, ],
['1-16-2019 3:00', 1, 16,	3, ], ['1-16-2019 4:00', 1, 16, 4, ], ['1-16-2019 5:00', 1, 16,	5, ], ['1-16-2019 6:00', 1, 16,	6, ], ['1-16-2019 7:00', 1, 16, 7, ], ['1-16-2019 8:00',	1, 16,	8, ], 
['1-16-2019 9:00', 1, 16,	9, ], ['1-16-2019 10:00', 1, 16, 10, ], ['1-16-2019 11:00', 1, 16, 11, ], ['1-16-2019 12:00', 1, 16, 12, ], ['1-16-2019 13:00:00', 1, 16, 13, ],
['1-16-2019 14:00:00', 1, 16,	14, ], ['1-16-2019 15:00:00', 1, 16, 15, ], ['1-16-2019 16:00:00', 1, 16, 16], ['1-16-2019 17:00:00', 1, 16, 17], ['1-16-2019 18:00:00', 1, 16, 10, ],
['1-16-2019 19:00:00', 1, 16, 19, ], ['1-16-2019 20:00:00', 1, 16, 20, ], ['1-16-2019 21:00:00', 1, 16, 21, ], ['1-16-2019 22:00:00',	1, 16, 22, ], ['1-16-2019 23:00:00', 1, 16, 23, ],
['1-17-2019 0:00', 1, 17, 0, ], ['1-17-2019 1:00', 1, 17, 1, ], ['1-17-2019 2:00', 1, 17, 2, ], ['1-17-2019 3:00', 1, 17,	3, ], ['1-17-2019 4:00', 1, 17, 4, ], ['1-17-2019 5:00', 1, 17,	5, ], 
['1-17-2019 6:00', 1, 17,	6, ], ['1-17-2019 7:00', 1, 17,	7, ], ['1-17-2019 8:00',	1, 17, 8, ], ['1-17-2019 9:00',	1, 17, 9, ], ['1-17-2019 10:00', 1,	17, 10, ], ['1-17-2019 11:00', 1,	17, 11, ], 
['1-17-2019 12:00', 1, 17, 12, ], ['1-17-2019 13:00:00', 1, 17,	13, ], ['1-17-2019 14:00:00', 1, 17, 14, ], ['1-17-2019 15:00:00',	1, 17, 15, ], ['1-17-2019 16:00:00', 1, 17, 16], 
['1-17-2019 17:00:00', 1, 17, 17], ['1-17-2019 18:00:00', 1, 17, 18, ], ['1-17-2019 19:00:00', 1, 17, 19, ], ['1-17-2019 20:00:00', 1, 17, 20, ], ['1-17-2019 21:00:00', 1, 17, 21, ],
['1-17-2019 22:00:00', 1, 17, 22, ], ['1-17-2019 23:00:00', 1, 17, 23, ], ['1-18-2019 0:00', 1, 18, 0, ], ['1-18-2019 1:00', 1, 18, 1, ], ['1-18-2019 2:00', 1, 18, 2, ],
['1-18-2019 3:00', 1, 18,	3, ], ['1-18-2019 4:00', 1, 18, 4, ], ['1-18-2019 5:00', 1, 18,	5, ], ['1-18-2019 6:00', 1, 18,	6, ], ['1-18-2019 7:00', 1, 18, 7, ], ['1-18-2019 8:00',	1, 18,	8, ], 
['1-18-2019 9:00', 1, 18,	9, ], ['1-18-2019 10:00', 1, 18, 10, ], ['1-18-2019 11:00', 1, 18, 11, ], ['1-18-2019 12:00', 1, 18, 12, ], ['1-18-2019 13:00:00', 1, 18, 13, ],
['1-18-2019 14:00:00', 1, 18,	14, ], ['1-18-2019 15:00:00', 1, 18, 15, ], ['1-18-2019 16:00:00', 1, 18, 16], ['1-18-2019 17:00:00', 1, 18, 17], ['1-18-2019 18:00:00', 1, 18, 10, ],
['1-18-2019 19:00:00', 1, 18, 19, ], ['1-18-2019 20:00:00', 1, 18, 20, ], ['1-18-2019 21:00:00', 1, 18, 21, ], ['1-18-2019 22:00:00',	1, 18, 22, ], ['1-18-2019 23:00:00', 1, 18, 23, ],
['1-19-2019 0:00', 1, 19, 0, ], ['1-19-2019 1:00', 1, 19, 1, ], ['1-19-2019 2:00', 1, 19, 2, ], ['1-19-2019 3:00', 1, 19,	3, ], ['1-19-2019 4:00', 1, 19, 4, ], ['1-19-2019 5:00', 1, 19,	5, ], 
['1-19-2019 6:00', 1, 19,	6, ], ['1-19-2019 7:00', 1, 19,	7, ], ['1-19-2019 8:00',	1, 19, 8, ], ['1-19-2019 9:00',	1, 19, 9, ], ['1-19-2019 10:00', 1,	19, 10, ], ['1-19-2019 11:00', 1,	19, 11, ], 
['1-19-2019 12:00', 1, 19, 12, ], ['1-19-2019 13:00:00', 1, 19,	13, ], ['1-19-2019 14:00:00', 1, 19, 14, ], ['1-19-2019 15:00:00',	1, 19, 15, ], ['1-19-2019 16:00:00', 1, 19, 16], 
['1-19-2019 17:00:00', 1, 19, 17], ['1-19-2019 18:00:00', 1, 19, 18, ], ['1-19-2019 19:00:00', 1, 19, 19, ], ['1-19-2019 20:00:00', 1, 19, 20, ], ['1-19-2019 21:00:00', 1, 19, 21, ],
['1-19-2019 22:00:00', 1, 19, 22, ], ['1-19-2019 23:00:00', 1, 19, 23, ], ['1-20-2019 0:00', 1, 20, 0, ], ['1-20-2019 1:00', 1, 20, 1, ], ['1-20-2019 2:00', 1, 20, 2, ],
['1-20-2019 3:00', 1, 20,	3, ], ['1-20-2019 4:00', 1, 20, 4, ], ['1-20-2019 5:00', 1, 20,	5, ], ['1-20-2019 6:00', 1, 20,	6, ], ['1-20-2019 7:00', 1, 20, 7, ], ['1-20-2019 8:00',	1, 20,	8, ], 
['1-20-2019 9:00', 1, 20,	9, ], ['1-20-2019 10:00', 1, 20, 10, ], ['1-20-2019 11:00', 1, 20, 11, ], ['1-20-2019 12:00', 1, 20, 12, ], ['1-20-2019 13:00:00', 1, 20, 13, ],
['1-20-2019 14:00:00', 1, 20,	14, ], ['1-20-2019 15:00:00', 1, 20, 15, ], ['1-20-2019 16:00:00', 1, 20, 16], ['1-20-2019 17:00:00', 1, 20, 17], ['1-20-2019 18:00:00', 1, 20, 10, ],
['1-20-2019 19:00:00', 1, 20, 19, ], ['1-20-2019 20:00:00', 1, 20, 20, ], ['1-20-2019 21:00:00', 1, 20, 21, ], ['1-20-2019 22:00:00',	1, 20, 22, ], ['1-20-2019 23:00:00', 1, 20, 23, ],
['1-21-2019 0:00', 1, 21, 0, ], ['1-21-2019 1:00', 1, 21, 1, ], ['1-21-2019 2:00', 1, 21, 2, ], ['1-21-2019 3:00', 1, 21,	3, ], ['1-21-2019 4:00', 1, 21, 4, ], ['1-21-2019 5:00', 1, 21,	5, ], 
['1-21-2019 6:00', 1, 21,	6, ], ['1-21-2019 7:00', 1, 21,	7, ], ['1-21-2019 8:00',	1, 21, 8, ], ['1-21-2019 9:00',	1, 21, 9, ], ['1-21-2019 10:00', 1,	21, 10, ], ['1-21-2019 11:00', 1,	21, 11, ], 
['1-21-2019 12:00', 1, 21, 12, ], ['1-21-2019 13:00:00', 1, 21,	13, ], ['1-21-2019 14:00:00', 1, 21, 14, ], ['1-21-2019 15:00:00',	1, 21, 15, ], ['1-21-2019 16:00:00', 1, 21, 16], 
['1-21-2019 17:00:00', 1, 21, 17], ['1-21-2019 18:00:00', 1, 21, 18, ], ['1-21-2019 19:00:00', 1, 21, 19, ], ['1-21-2019 20:00:00', 1, 21, 20, ], ['1-21-2019 21:00:00', 1, 21, 21, ],
['1-21-2019 22:00:00', 1, 21, 22, ], ['1-21-2019 23:00:00', 1, 21, 23, ], ['1-22-2019 0:00', 1, 22, 0, ], ['1-22-2019 1:00', 1, 22, 1, ], ['1-22-2019 2:00', 1, 22, 2, ],
['1-22-2019 3:00', 1, 22,	3, ], ['1-22-2019 4:00', 1, 22, 4, ], ['1-22-2019 5:00', 1, 22,	5, ], ['1-22-2019 6:00', 1, 22,	6, ], ['1-22-2019 7:00', 1, 22, 7, ], ['1-22-2019 8:00',	1, 22,	8, ], 
['1-22-2019 9:00', 1, 22,	9, ], ['1-22-2019 10:00', 1, 22, 10, ], ['1-22-2019 11:00', 1, 22, 11, ], ['1-22-2019 12:00', 1, 22, 12, ], ['1-22-2019 13:00:00', 1, 22, 13, ],
['1-22-2019 14:00:00', 1, 22,	14, ], ['1-22-2019 15:00:00', 1, 22, 15, ], ['1-22-2019 16:00:00', 1, 22, 16], ['1-22-2019 17:00:00', 1, 22, 17], ['1-22-2019 18:00:00', 1, 22, 10, ],
['1-22-2019 19:00:00', 1, 22, 19, ], ['1-22-2019 20:00:00', 1, 22, 20, ], ['1-22-2019 21:00:00', 1, 22, 21, ], ['1-22-2019 22:00:00',	1, 22, 22, ], ['1-22-2019 23:00:00', 1, 22, 23, ],
['1-23-2019 0:00', 1, 23, 0, ], ['1-23-2019 1:00', 1, 23, 1, ], ['1-23-2019 2:00', 1, 23, 2, ], ['1-23-2019 3:00', 1, 23,	3, ], ['1-23-2019 4:00', 1, 23, 4, ], ['1-23-2019 5:00', 1, 23,	5, ], 
['1-23-2019 6:00', 1, 23,	6, ], ['1-23-2019 7:00', 1, 23,	7, ], ['1-23-2019 8:00',	1, 23, 8, ], ['1-23-2019 9:00',	1, 23, 9, ], ['1-23-2019 10:00', 1,	23, 10, ], ['1-23-2019 11:00', 1,	23, 11, ], 
['1-23-2019 12:00', 1, 23, 12, ], ['1-23-2019 13:00:00', 1, 23,	13, ], ['1-23-2019 14:00:00', 1, 23, 14, ], ['1-23-2019 15:00:00',	1, 23, 15, ], ['1-23-2019 16:00:00', 1, 23, 16], 
['1-23-2019 17:00:00', 1, 23, 17], ['1-23-2019 18:00:00', 1, 23, 18, ], ['1-23-2019 19:00:00', 1, 23, 19, ], ['1-23-2019 20:00:00', 1, 23, 20, ], ['1-23-2019 21:00:00', 1, 23, 21, ],
['1-23-2019 22:00:00', 1, 23, 22, ], ['1-23-2019 23:00:00', 1, 23, 23, ], ['1-24-2019 0:00', 1, 24, 0, ], ['1-24-2019 1:00', 1, 24, 1, ], ['1-24-2019 2:00', 1, 24, 2, ],
['1-24-2019 3:00', 1, 24,	3, ], ['1-24-2019 4:00', 1, 24, 4, ], ['1-24-2019 5:00', 1, 24,	5, ], ['1-24-2019 6:00', 1, 24,	6, ], ['1-24-2019 7:00', 1, 24, 7, ], ['1-24-2019 8:00',	1, 24,	8, ], 
['1-24-2019 9:00', 1, 24,	9, ], ['1-24-2019 10:00', 1, 24, 10, ], ['1-24-2019 11:00', 1, 24, 11, ], ['1-24-2019 12:00', 1, 24, 12, ], ['1-24-2019 13:00:00', 1, 24, 13, ],
['1-24-2019 14:00:00', 1, 24,	14, ], ['1-24-2019 15:00:00', 1, 24, 15, ], ['1-24-2019 16:00:00', 1, 24, 16], ['1-24-2019 17:00:00', 1, 24, 17], ['1-24-2019 18:00:00', 1, 24, 10, ],
['1-24-2019 19:00:00', 1, 24, 19, ], ['1-24-2019 20:00:00', 1, 24, 20, ], ['1-24-2019 21:00:00', 1, 24, 21, ], ['1-24-2019 22:00:00',	1, 24, 22, ], ['1-24-2019 23:00:00', 1, 24, 23, ],
['1-25-2019 0:00', 1, 25, 0, ], ['1-25-2019 1:00', 1, 25, 1, ], ['1-25-2019 2:00', 1, 25, 2, ], ['1-25-2019 3:00', 1, 25,	3, ], ['1-25-2019 4:00', 1, 25, 4, ], ['1-25-2019 5:00', 1, 25,	5, ], 
['1-25-2019 6:00', 1, 25,	6, ], ['1-25-2019 7:00', 1, 25,	7, ], ['1-25-2019 8:00',	1, 25, 8, ], ['1-25-2019 9:00',	1, 25, 9, ], ['1-25-2019 10:00', 1,	25, 10, ], ['1-25-2019 11:00', 1,	25, 11, ], 
['1-25-2019 12:00', 1, 25, 12, ], ['1-25-2019 13:00:00', 1, 25,	13, ], ['1-25-2019 14:00:00', 1, 25, 14, ], ['1-25-2019 15:00:00',	1, 25, 15, ], ['1-25-2019 16:00:00', 1, 25, 16], 
['1-25-2019 17:00:00', 1, 25, 17], ['1-25-2019 18:00:00', 1, 25, 18, ], ['1-25-2019 19:00:00', 1, 25, 19, ], ['1-25-2019 20:00:00', 1, 25, 20, ], ['1-25-2019 21:00:00', 1, 25, 21, ],
['1-25-2019 22:00:00', 1, 25, 22, ], ['1-25-2019 23:00:00', 1, 25, 23, ], ['1-26-2019 0:00', 1, 26, 0, ], ['1-26-2019 1:00', 1, 26, 1, ], ['1-26-2019 2:00', 1, 26, 2, ],
['1-26-2019 3:00', 1, 26,	3, ], ['1-26-2019 4:00', 1, 26, 4, ], ['1-26-2019 5:00', 1, 26,	5, ], ['1-26-2019 6:00', 1, 26,	6, ], ['1-26-2019 7:00', 1, 26, 7, ], ['1-26-2019 8:00',	1, 26,	8, ], 
['1-26-2019 9:00', 1, 26,	9, ], ['1-26-2019 10:00', 1, 26, 10, ], ['1-26-2019 11:00', 1, 26, 11, ], ['1-26-2019 12:00', 1, 26, 12, ], ['1-26-2019 13:00:00', 1, 26, 13, ],
['1-26-2019 14:00:00', 1, 26,	14, ], ['1-26-2019 15:00:00', 1, 26, 15, ], ['1-26-2019 16:00:00', 1, 26, 16], ['1-26-2019 17:00:00', 1, 26, 17], ['1-26-2019 18:00:00', 1, 26, 10, ],
['1-26-2019 19:00:00', 1, 26, 19, ], ['1-26-2019 20:00:00', 1, 26, 20, ], ['1-26-2019 21:00:00', 1, 26, 21, ], ['1-26-2019 22:00:00',	1, 26, 22, ], ['1-26-2019 23:00:00', 1, 26, 23, ],
['1-27-2019 0:00', 1, 27, 0, ], ['1-27-2019 1:00', 1, 27, 1, ], ['1-27-2019 2:00', 1, 27, 2, ], ['1-27-2019 3:00', 1, 27,	3, ], ['1-27-2019 4:00', 1, 27, 4, ], ['1-27-2019 5:00', 1, 27,	5, ], 
['1-27-2019 6:00', 1, 27,	6, ], ['1-27-2019 7:00', 1, 27,	7, ], ['1-27-2019 8:00',	1, 27, 8, ], ['1-27-2019 9:00',	1, 27, 9, ], ['1-27-2019 10:00', 1,	27, 10, ], ['1-27-2019 11:00', 1,	27, 11, ], 
['1-27-2019 12:00', 1, 27, 12, ], ['1-27-2019 13:00:00', 1, 27,	13, ], ['1-27-2019 14:00:00', 1, 27, 14, ], ['1-27-2019 15:00:00',	1, 27, 15, ], ['1-27-2019 16:00:00', 1, 27, 16], 
['1-27-2019 17:00:00', 1, 27, 17], ['1-27-2019 18:00:00', 1, 27, 18, ], ['1-27-2019 19:00:00', 1, 27, 19, ], ['1-27-2019 20:00:00', 1, 27, 20, ], ['1-27-2019 21:00:00', 1, 27, 21, ],
['1-27-2019 22:00:00', 1, 27, 22, ], ['1-27-2019 23:00:00', 1, 27, 23, ], ['1-28-2019 0:00', 1, 28, 0, ], ['1-28-2019 1:00', 1, 28, 1, ], ['1-28-2019 2:00', 1, 28, 2, ],
['1-28-2019 3:00', 1, 28,	3, ], ['1-28-2019 4:00', 1, 28, 4, ], ['1-28-2019 5:00', 1, 28,	5, ], ['1-28-2019 6:00', 1, 28,	6, ], ['1-28-2019 7:00', 1, 28, 7, ], ['1-28-2019 8:00',	1, 28,	8, ], 
['1-28-2019 9:00', 1, 28,	9, ], ['1-28-2019 10:00', 1, 28, 10, ], ['1-28-2019 11:00', 1, 28, 11, ], ['1-28-2019 12:00', 1, 28, 12, ], ['1-28-2019 13:00:00', 1, 28, 13, ],
['1-28-2019 14:00:00', 1, 28,	14, ], ['1-28-2019 15:00:00', 1, 28, 15, ], ['1-28-2019 16:00:00', 1, 28, 16], ['1-28-2019 17:00:00', 1, 28, 17], ['1-28-2019 18:00:00', 1, 28, 10, ],
['1-28-2019 19:00:00', 1, 28, 19, ], ['1-28-2019 20:00:00', 1, 28, 20, ], ['1-28-2019 21:00:00', 1, 28, 21, ], ['1-28-2019 22:00:00',	1, 28, 22, ], ['1-28-2019 23:00:00', 1, 28, 23, ],
['1-29-2019 0:00', 1, 29, 0, ], ['1-29-2019 1:00', 1, 29, 1, ], ['1-25-2019 2:00', 1, 29, 2, ], ['1-29-2019 3:00', 1, 29,	3, ], ['1-29-2019 4:00', 1, 29, 4, ], ['1-25-2019 5:00', 1, 25,	5, ], 
['1-29-2019 6:00', 1, 29,	6, ], ['1-29-2019 7:00', 1, 29,	7, ], ['1-25-2019 8:00',	1, 29, 8, ], ['1-29-2019 9:00',	1, 29, 9, ], ['1-29-2019 10:00', 1,	29, 10, ], ['1-25-2019 11:00', 1,	25, 11, ], 
['1-29-2019 12:00', 1, 29, 12, ], ['1-29-2019 13:00:00', 1, 29,	13, ], ['1-25-2019 14:00:00', 1, 29, 14, ], ['1-29-2019 15:00:00',	1, 29, 15, ], ['1-29-2019 16:00:00', 1, 29, 16], 
['1-29-2019 17:00:00', 1, 29, 17], ['1-29-2019 18:00:00', 1, 29, 18, ], ['1-25-2019 19:00:00', 1, 29, 19, ], ['1-29-2019 20:00:00', 1, 29, 20, ], ['1-29-2019 21:00:00', 1, 29, 21, ],
['1-29-2019 22:00:00', 1, 29, 22, ], ['1-29-2019 23:00:00', 1, 29, 23, ], ['1-30-2019 0:00', 1, 30, 0, ], ['1-30-2019 1:00', 1, 30, 1, ], ['1-30-2019 2:00', 1, 30, 2, ],
['1-30-2019 3:00', 1, 30,	3, ], ['1-30-2019 4:00', 1, 30, 4, ], ['1-30-2019 5:00', 1, 30,	5, ], ['1-30-2019 6:00', 1, 30,	6, ], ['1-30-2019 7:00', 1, 30, 7, ], ['1-30-2019 8:00',	1, 30,	8, ], 
['1-30-2019 9:00', 1, 30,	9, ], ['1-30-2019 10:00', 1, 30, 10, ], ['1-30-2019 11:00', 1, 30, 11, ], ['1-30-2019 12:00', 1, 30, 12, ], ['1-30-2019 13:00:00', 1, 30, 13, ],
['1-30-2019 14:00:00', 1, 30,	14, ], ['1-30-2019 15:00:00', 1, 30, 15, ], ['1-30-2019 16:00:00', 1, 30, 16], ['1-30-2019 17:00:00', 1, 30, 17], ['1-30-2019 18:00:00', 1, 30, 10, ],
['1-30-2019 19:00:00', 1, 30, 19, ], ['1-30-2019 20:00:00', 1, 30, 20, ], ['1-30-2019 21:00:00', 1, 30, 21, ], ['1-30-2019 22:00:00',	1, 30, 22, ], ['1-30-2019 23:00:00', 1, 30, 23, ],
['1-31-2019 0:00', 1, 31, 0, ], ['1-31-2019 1:00', 1, 31, 1, ], ['1-31-2019 2:00', 1, 31, 2, ], ['1-31-2019 3:00', 1, 31,	3, ], ['1-31-2019 4:00', 1, 31, 4, ], ['1-31-2019 5:00', 1, 31,	5, ], 
['1-31-2019 6:00', 1, 31,	6, ], ['1-31-2019 7:00', 1, 31,	7, ], ['1-31-2019 8:00',	1, 31, 8, ], ['1-31-2019 9:00',	1, 31, 9, ], ['1-31-2019 10:00', 1,	31, 10, ], ['1-31-2019 11:00', 1,	31, 11, ], 
['1-31-2019 12:00', 1, 31, 12, ], ['1-31-2019 13:00:00', 1, 31,	13, ], ['1-31-2019 14:00:00', 1, 31, 14, ], ['1-31-2019 15:00:00',	1, 31, 15, ], ['1-31-2019 16:00:00', 1, 31, 16], 
['1-31-2019 17:00:00', 1, 31, 17], ['1-31-2019 18:00:00', 1, 31, 18, ], ['1-31-2019 19:00:00', 1, 31, 19, ], ['1-31-2019 20:00:00', 1, 31, 20, ], ['1-31-2019 21:00:00', 1, 31, 21, ],
['1-31-2019 22:00:00', 1, 31, 22, ], ['1-31-2019 23:00:00', 1, 31, 23, ], 

df = pd.DataFrame(data2, columns=['Timestamp', 'month', 'day', 'hour', 'trips')

pred = modelFit.predict(data2)
