# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
#import os
#import numpy as np
import matplotlib.pyplot as plt


'''
#REDUCED SYNTAX:
classifier = Sequential([
    Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Conv2D(32, (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    Flatten(),
    Dense(units = 128, activation = 'relu'),
    Dense(units = 1, activation='sigmoid')
])
classifier.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
classifier.summary()
#___________________
'''
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2)) #Adding Dropout to reduce overfitting (randomly kills 20% of the output units in each training epoch)

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu')) #original 32 features map
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.3)) #Adding Dropout to reduce overfitting (randomly kills 20% of the output units in each training epoch)
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
'''#Improved model with lower learning rate
model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.0001),
            metrics=['accuracy'])
'''
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

epochs=25 #original = 25

#IMAGE AUGMENTATION (generating more training data using random tranformation),
#to beat overfitting: small number of training examples
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#original batch size = 32
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 shuffle=True,
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'binary')

#original batch size = 32
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'binary')

#OPCIONAL_____________________________________________________________________
'''
#visualize training images
sample_training_images, _ = next(training_set)
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])
'''
#_____________________________________________________________________________

#original epochs = 25
history = classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = epochs,
                         #use_multiprocessing=True,
                         workers=4,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save('img_model.h5')

#RESULTS AND PERFORMANCE:
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#USING THE MODEL ON NEW IMAGES
from keras.models import load_model
import cv2
import numpy as np

model = load_model('img_model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def prediccion(img):
  plt.imshow(img)
  img2 = cv2.resize(img,(64,64))
  img2 = np.reshape(img2,[1,64,64,3])
  classes = model.predict_classes(img2)
  if classes == 0: 
    classes = "Cat"
    else:
    classes = "Dog"
  return (classes)

imagen = cv2.imread('dataset/prediction_img/test6.jpg')
prediccion(imagen)


#solution from instructor
from keras.preprocessing import image
test_image = image.load_img('dataset/prediction_img/test2.jpg', target_size = (64,64))
test_image2 = image.img_to_array(test_image)
test_image2 = np.expand_dims(test_image2,axis = 0)
training_set.class_indices
result = model.predict(test_image2)
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
plt.imshow(test_image)
prediction