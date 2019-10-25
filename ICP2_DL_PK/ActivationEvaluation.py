# Importing libraries
from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

# Loading input data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display the first image in the training data
plt.imshow(train_images[0, :, :], cmap='gray')
plt.title('Ground Truth : {}'.format(train_labels[0]))
plt.show()

# Process the data
# 1. Convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# Convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')

# Scale data
train_data /=255.0
test_data /=255.0

# Change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Creating network
model = Sequential()

# Relu function ranges the data to fall between 0 and z (the value of range present)
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(400, activation='sigmoid'))
model.add(Dense(300, activation='tanh'))
model.add(Dense(10, activation='softmax'))

# # Sigmoid function ranges the data to fall between 0 and 1
# model.add(Dense(512, activation='sigmoid', input_shape=(dimData,)))
# model.add(Dense(400, activation='sigmoid'))
# model.add(Dense(250, activation='sigmoid'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(150, activation='tanh'))
# model.add(Dense(10, activation='softmax'))
#
# # Tangent (tanh) function ranges the data to fall between -1 and 1
# model.add(Dense(512, activation='tanh', input_shape=(dimData,)))
# model.add(Dense(300, activation='tanh'))
# model.add(Dense(150, activation='tanh'))
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(250, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1, validation_data=(test_data, test_labels_one_hot))
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))