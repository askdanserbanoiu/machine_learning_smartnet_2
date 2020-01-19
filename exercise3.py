import os
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
from skimage.transform import resize
import matplotlib.pyplot as plt


def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        image_dir=os.path.join(folder, filename)
        #image_dir = mpimg.imread(os.path.join(folder, filename))
        if image_dir is not None:
            images.append(image_dir)
            # load & smoothen image
            kernel = np.ones((7,7),np.float32)/49
            image = cv.imread(image_dir,cv.IMREAD_GRAYSCALE)
            image = cv.filter2D(image,-1,kernel)
            # make numpy array
            image = np.array(image)
            image = resize(image, (28,28))
            # make negative
            image = np.ones(image.shape) - image
            plt.imshow(image, cmap="gray")
            plt.show()
    return images
     

imgs = load_images('digits')

print (imgs)

"""
image, cmap="gray")
plt.show()

# Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 

# Define model architecture
model = Sequential()
"""

"""
Because of error:
-------------------------------------------------------------------------------- 
ValueError: Negative dimension size caused by subtracting 3 from 1 for 
'conv2d_2/convolution' (op: 'Conv2D') with input shapes: [?,1,28,28],[3,3,28,32].
--------------------------------------------------------------------------------
 input_shape=(1,28,28),data_format='channels_first' is added
"""
"""
model.add(Convolution2D(9, (3, 3), padding='same', activation='relu', input_shape=(1,28,28),data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(36, (3, 3), padding='same', activation='relu',  data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Convolution2D(49, (3, 3), padding='same', activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(3,3)))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
 
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=2, verbose=1)
 
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
"""