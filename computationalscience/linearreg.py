import numpy as np
import matplotlib.pyplot as plt

# Function to input x and y values from the user
def get_input_data():
    n = int(input("How many data points you want to enter?: "))
    x = []
    y = []
    for i in range(n):
        x_value = float(input(f"Enter data point x[{i+1}]: "))
        y_value = float(input(f"Enter data point y[{i+1}]: "))
        x.append(x_value)
        y.append(y_value)
    return np.array(x), np.array(y)

# Get input data from the user
x, y = get_input_data()

# formula for linear regression (y = mx + b):
# Formula for m (slope) and b (intercept):
# m = Σ((x_i - x_mean)(y_i - y_mean)) / Σ((x_i - x_mean)^2)
# b = y mean - m * x mean

# means of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# slope (m)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean)**2)
m = numerator / denominator

# intercept (b)
b = y_mean - m * x_mean

# linear regression equation
print(f"Linear Regression Equation: y = {m:.2f}x + {b:.2f}")

# Plot of the original data points and the regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, m * x + b, color='red', label=f'Linear regression: y = {m:.2f}x + {b:.2f}')
plt.title('Manual Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
