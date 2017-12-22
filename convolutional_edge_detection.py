from keras.models import Sequential
from keras.layers import (Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D)
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import json, pylab
import mahotas as mh
import numpy as np

np.set_printoptions(threshold=np.nan)



# some model and data processing constants
batch_size = 128
nb_classes = 2
nb_epoch = 7

# input image dimensions
img_rows, img_cols = 28, 28

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# architecture
model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# retrieve the data
train_imgs = []
ground_truth_train = []
train_imgs.append(mh.colors.rgb2grey(mh.imread('2.jpg'),dtype = np.float))
ground_truth_train.append(mh.imread('2_edge.jpg'))
train_imgs.append(mh.colors.rgb2grey(mh.imread('3.jpg'),dtype = np.float))
ground_truth_train.append(mh.colors.rgb2grey(mh.imread('3_edge.jpg')))
train_imgs.append(mh.colors.rgb2grey(mh.imread('4.jpg'),dtype = np.float))
ground_truth_train.append(mh.colors.rgb2grey(mh.imread('4_edge.jpg')))

test_imgs = []
ground_truth_test = []
test_imgs.append(mh.colors.rgb2grey(mh.imread('normal.jpg'),dtype = np.float))
ground_truth_test.append(mh.imread('normal_edge.jpg'))
test_imgs.append(mh.colors.rgb2grey(mh.imread('cloudy.jpg'),dtype = np.float))
ground_truth_test.append(mh.colors.rgb2grey(mh.imread('cloudy_edge.jpg')))

X_train, X_test, y_train, y_test = [], [], [], []
print('Preparing images...')
for x in range(img_rows//2,len(train_imgs[0])-img_rows//2):
    for y in range(img_cols//2,len(train_imgs[0][0])-img_cols//2):
        for i in range(len(train_imgs)):
            if(ground_truth_train[i][x][y] > 0 or (x+y)%30000 == 0):
                X_train.append(train_imgs[i][x-img_rows//2:x+img_rows//2,y-img_cols//2:y+img_cols//2])
                y_train.append(ground_truth_train[i][x][y])
        for i in range(len(test_imgs)):
            if(ground_truth_test[i][x][y] > 0 or (x+y)%30000 == 0):
                X_test.append(test_imgs[i][x-img_rows//2:x+img_rows//2,y-img_cols//2:y+img_cols//2])
                y_test.append(ground_truth_test[i][x][y])
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# prepare the data
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
print(X_train.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np.divide(y_train,255)
y_test = np.divide(y_test,255)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# train it plz
print('Training model...')
model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch, verbose=1,
          validation_data=(X_test, Y_test))

# let's dump the model
print('Saving model...')
saved_model = model.to_json()
with open('CNN_architecture.json', 'w') as outfile:
    json.dump(saved_model, outfile)
model.save_weights('CNN_weights.h5')
