# Sign Language Classification using Convolutional Neural Network (CNN)
This repository contains Python code for building a Convolutional Neural Network (CNN) to classify sign language gestures. The model is trained on the "Sign Language MNIST" dataset, which consists of images of sign language gestures representing letters from A to Z (excluding J and Z).

# Dependencies
Ensure you have the following libraries installed:

TensorFlow
Keras
Matplotlib
NumPy
Pandas
OpenCV (for data visualization)
Install the required libraries using pip:

pip install tensorflow keras matplotlib numpy pandas opencv-python

# Dataset
The "Sign Language MNIST" dataset contains grayscale images of sign language gestures. The training set consists of 27,455 images, and the test set contains 7,172 images. Each image is of size 28x28 pixels.

# Code Overview
Data Preprocessing: The code loads the dataset and preprocesses the images by scaling their pixel values to the range [0, 1].

Visualization: Various functions are provided for visualizing the dataset, including distribution graphs (histograms/bar graphs) of column data and correlation matrix plots.

CNN Model: A CNN model is defined using the TensorFlow-Keras API. The model consists of convolutional layers, max-pooling layers, and dense (fully connected) layers. The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

Training: The model is trained on the training set for 10 epochs with a batch size of 128.

Evaluation: The trained model is evaluated on the test set to measure its performance. The accuracy and loss metrics are displayed.

# Running the Code
Download the "Sign Language MNIST" dataset from Kaggle and place the sign_mnist_train.csv and sign_mnist_test.csv files in the data/ directory.

Open the Jupyter Notebook or Python script containing the provided code.

Run the code cells or execute the script.

# Note
This code uses a simple CNN architecture for demonstration purposes. Depending on your specific requirements, you can experiment with different CNN architectures or consider using more advanced models like VGG, ResNet, etc., for potentially better accuracy.

Feel free to modify the code to suit your needs or build upon it for more advanced sign language recognition applications. Happy coding!
