# Import libraries
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt

# Size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# Input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# Mapping input to its reconstruction
autoencoder = Model(input_img, decoded)
# Model mapping an input to its encoded representation
encoder = Model(input_img, encoded)
# Creating a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# Retrieving the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Creating the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
# Compiling the model defined
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])

# Loading input data set
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Visualization of the results (accuracy, loss) using Tensor board
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Fit the model using tensor board
history = autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=25, batch_size=256, callbacks=[tbCallBack])

[test_loss, test_acc] = autoencoder.evaluate(x_test, x_test)
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