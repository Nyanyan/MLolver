import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
allData = pd.read_csv('data.csv')

plt.scatter(allData['cp'],allData['num'])
plt.show()

X1 = allData[['cp']]
Y1 = allData['num']
model1 = LinearRegression()
model1.fit(X1, Y1)

plt.scatter(allData['cp'],allData['num'])
plt.plot(X1, model1.predict(X1))
plt.show()

model1.predict(4000)

print(model1.coef_)
print(model1.intercept_)
print(model1.score(X1, Y1))