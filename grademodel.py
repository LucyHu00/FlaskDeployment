import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('grade.csv')

X = dataset.iloc[:, 1:3]
y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('newmodel.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('newmodel.pkl','rb'))
print(model.predict([[23, 165]]))