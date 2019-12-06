# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

# Reading data
deathData = pd.read_csv('DeathRate.csv')

# Identifying features and predictor variables associated with the heart data set
x = deathData.iloc[:, 1:16]
y = deathData.iloc[:, 16]

# Split the data set into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
print(x_train.size)
print(x_train.shape)

# Creating neural network model for death rate analysis
# Define the model to be generated/built
nnDeath = Sequential()
# Provide input and neurons for first hidden dense layer
nnDeath.add(Dense(15, input_dim=15, activation='relu'))
# Define the output neuron
nnDeath.add(Dense(1, activation='sigmoid'))

# Fitting the neural network model on the training data set
nnDeath.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = nnDeath.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0, initial_epoch=0)

# Evaluation of the loss and accuracy associated to the test data set
[test_loss, test_acc] = nnDeath.evaluate(x_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Listing all the components of data present in history
print('The data components present in history are', history.history.keys())

# Graphical evaluation of accuracy associated with training and validation data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Evaluation of Data Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy of Data')
plt.legend(['TrainData', 'ValidationData'], loc='upper right')
plt.show()

# Graphical evaluation of loss associated with training and validation data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('Loss of Data')
plt.title('Evaluation of Data Loss')
plt.legend(['TrainData', 'ValidationData'], loc='upper right')
plt.show()
