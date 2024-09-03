import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
np.random.seed(0)
num_samples = 100
square_footage = np.random.rand(num_samples) * 2000  
num_bedrooms = np.random.randint(1, 5, num_samples)  
num_bathrooms = np.random.randint(1, 4, num_samples) 
prices = (square_footage * 300 + num_bedrooms * 5000 + num_bathrooms * 3000 + np.random.randn(num_samples) * 10000)  
data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Bedrooms': num_bedrooms,
    'Bathrooms': num_bathrooms,
    'Price': prices
})
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()