# Importing libraries
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Reading data
breastCancer = pd.read_csv('BreastCancer.csv')
print(breastCancer)

# Converting non-numerical data into numerical
breastCancer["diagnosis"] = pd.Categorical(breastCancer["diagnosis"])
breastCancer["diagnosis"] = breastCancer["diagnosis"].cat.codes
cancerData = breastCancer.values

# Split the data set into training and test sets
x_train, x_test, y_train, y_test = train_test_split(cancerData[:, 2:32], cancerData[:, 1], test_size=0.2, random_state=45)
print(x_train.size)
print(x_train.shape)

# Normalize training data set using standard scaler
sc = StandardScaler()
scData = sc.fit(x_train, x_test)
print(scData)

# Creating neural network model for breast cancer diagnosis
# Define the model to be generated/built
nnCancer = Sequential()
# Provide input and neurons for first hidden dense layer
nnCancer.add(Dense(15, input_dim=30, activation='relu'))
# Define the output neuron
nnCancer.add(Dense(1, activation='sigmoid'))

# Fitting the neural network model on the training data set
nnCancer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nnCancerModel = nnCancer.fit(x_train, y_train, epochs=100, verbose=0, initial_epoch=0)

# Display the neural network identified
print('The summary of the neural network is', nnCancer.summary())
print(nnCancer.evaluate(x_test, y_test, verbose=0))

