#train

from google.colab import drive
drive.mount('/content/drive')
##Importing Libararies

import tensorflow as tf
import matplotlib.pyplot as plt

#Data Preprocessing
#Training Image preporcessing
training_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/archive (1)/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

#Validation Image Preprocessing
validation_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/archive (1)/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

#Building Model
cnn = tf.keras.models.Sequential()

#Building Convolution Layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input

# Initialize the model
cnn = Sequential()

# Define the input layer
cnn.add(Input(shape=(64, 64, 3)))

# Add the convolutional layer
cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))

# Add the max pooling layer
cnn.add(MaxPool2D(pool_size=2, strides=2))


cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.5)) #To avoid overfitting
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

#Output layer
cnn.add(tf.keras.layers.Dense(units=36,activation='softmax'))

#Compiling and Training Phase
cnn.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
training_history= cnn.fit(x=training_set,validation_data=validation_set,epochs=30)

#Saving Model
cnn.save('trained_model.keras')
training_history.history #Return Dictionary of history

#Recording History in json
import json
with open('training_hist.json', 'w') as f:
    json.dump(training_history.history, f)
print(training_history.history.keys())

##Calculating Accuracy of Model Archieved on Validation Set
print("Validation set Accuracy: {} &" .format(training_history.history['val_accuracy'][-1]*100))

#Accuracy Visulization
##Training Visulization
#training_history.history['accuracy']
epochs = [i for i in range(1,31)]
plt.plot(epochs,training_history.history['accuracy'],color='red')
plt.xlabel('No. of Epochs')
plt.ylabel('Training Accuracy')
plt.title('Visualization of Training Accuracy Result')
plt.show()
epochs

##Validation Accuracy
plt.plot(epochs,training_history.history['val_accuracy'],color='blue')
plt.xlabel('No. of Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Visualization of Validation Accuracy Result')
plt.show()
