#test

from google.colab import drive
drive.mount('/content/drive')

##Importing libararies
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Connect to drive
from google.colab import drive
drive.mount("/content/drive")

##Loding Model
cnn = tf.keras.models.load_model("/content/trained_model.keras")

##Visulization and Performing Prediction on single image
import cv2
image_path = "/content/drive/MyDrive/archive (1)/test/cauliflower/Image_1.jpg"
img = cv2.imread(image_path)
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()

##Testing Model
image=tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = cnn.predict(input_arr)
print(predictions[0])
print(max(predictions[0]))
test_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/archive (1)/test',
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
test_set.class_names
result_index = np.where(predictions[0] == max(predictions[0]))
print(result_index)

#Display Image
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()

#Single Predictions
print("It's a {}".format(test_set.class_names[result_index[0][0]]))