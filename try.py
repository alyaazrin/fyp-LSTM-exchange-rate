# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Reading CSV file into training set
training_set = pd.read_csv('alya_formatted_2.csv')
training_set.head()

dataset = read_csv('alya_formatted_2.csv', header=0, index_col=0)
values = dataset.values

n_days_predict=5
n_features=5

# Reading CSV file into test set
test_set = pd.read_csv('alya_formatted_2.csv')
test_set.head()

# Getting relevant feature
training_set = training_set.iloc[:,1:2]
training_set.head()

# Converting to 2D array
training_set = training_set.values
training_set

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
training_set

# Getting the inputs and the ouputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Example
today = pd.DataFrame(X_train[0:5])
tomorrow = pd.DataFrame(y_train[0:5])
ex = pd.concat([today, tomorrow], axis=1)
ex.columns = (['today', 'tomorrow'])
ex

# Reshaping into required shape for Keras
X_train = np.reshape(X_train, (1257, 1, 1))
X_train

# Initializing the Recurrent Neural Network
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Compiling the Recurrent Neural Network
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the Recurrent Neural Network to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 5)

# Getting the real stock price of 
test_set = pd.read_csv('alya_formatted_2.csv')
test_set.head()

# Getting relevant feature
real_stock_price = test_set.iloc[:,1:2]
real_stock_price.head()

# Converting to 2D array
real_stock_price = real_stock_price.values

# Getting the predicted stock price of
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (168, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real USD-MYR Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted USD-MYR Price')
plt.title('USD-MYR Price Prediction')
plt.xlabel('Days')
plt.ylabel('USD-MYR Price')
plt.legend()
plt.show()



 # If 'q' is pressed, close program
if 0xFF == ord('q'):
	sys.exit()


