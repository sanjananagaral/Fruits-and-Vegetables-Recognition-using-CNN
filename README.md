# Fruits-and-Vegetables-Recognition-using-CNN

CNN-Based Fruit and Vegetable Recognition
This project uses a deep learning model to identify and categorise various fruits and vegetables from pictures.
It is trained on a Kaggle dataset in Google Colab and constructed with Python and TensorFlow/Keras.

Project Files 
`train.py` → For training the model 
`test.py` → For testing and image prediction 
`trained_model.keras` → Saved trained model 
`training_hist.json` → Stores training history

How It Operates
1. Pictures of fruits and vegetables are used to train the CNN model.
2. The model can identify which fruit or vegetable is in a given image once it has been trained.
3. Both the image and its anticipated label are displayed in the output.

Tools and Libraries:
Matlotlib 
OpenCV 
TensorFlow/Keras 
NumPy 
Google Colab

Model Details:
* Input size: 64x64 pixels
* 2 Convolution + MaxPooling layers
* Dropout layer to reduce overfitting
* Dense layer with ReLU activation
* Output layer with Softmax (36 classes)
