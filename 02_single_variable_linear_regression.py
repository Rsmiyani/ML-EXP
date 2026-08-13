import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
marks = np.array([32, 39, 45, 52, 60, 65, 73, 80])
model = LinearRegression()
model.fit(hours, marks)
print('Slope:', model.coef_[0])
print('Intercept:', model.intercept_)
print('Predicted marks for 6 hours:', model.predict([[6]])[0])
plt.scatter(hours.ravel(), marks)
plt.plot(hours.ravel(), model.predict(hours), color='red')
plt.xlabel('Study hours')
plt.ylabel('Marks')
plt.title('Simple Linear Regression')
plt.show()
