import keras

from utils import load_data
from my_CNN_model import *
import cv2
from keras.datasets import mnist
from keras import backend as K

num_classes = 10


img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Load training set
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# NOTE: Please check the load_data() method in utils.py to see how the data is preprocessed (normalizations and stuff)


# Setting the CNN architecture
my_model = get_my_CNN_model_architecture()

# Compiling the CNN model with an appropriate optimizer and loss and metrics
compile_my_CNN_model(my_model, optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training the model
hist = train_my_CNN_model(my_model, x_train, y_train)

# train_my_CNN_model returns a History object. History.history attribute is a record of training loss values and metrics
# values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

# Saving the model
save_my_CNN_model(my_model, 'my_model')

'''
# You can skip all the steps above (from 'Setting the CNN architecture') after running the script for the first time.
# Just load the recent model using load_my_CNN_model and use it to predict keypoints on any face data
my_model = load_my_CNN_model('my_model')
'''
