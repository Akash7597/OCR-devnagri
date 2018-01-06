# About
An Optical Character Recognition(OCR) for the devanagri script
using Convolutional Neural Networks(CNNs).

# Requirements
The image may contain more than one character. 
Therefore this is a case of multi-label multi-class classification.

# Neural-Net Architecture
1) The Output layer of net contains 128 neurons corresponding to each
devanagri character.
2) The input layer is a Convolutional layer taking images of dimension
64*64
3) The hidden layer consists of Convlational layers and Max-Pooling layers with
a ReLu activation function
4) A sigmoid activation function is used in the last layer, with a threshold value of 0.5 per character. Softmax isn't
suited for multi-label classification([read](https://stackoverflow.com/questions/44164749/how-does-keras-handle-multilabel-classification)).

# Preprocessing
1) A Gaussian blur filter is used for noise removal.
2) Binarization - Images are converted to grayscale then OTSU's method is 
used for grayscale to binary image conversion.
3) The images are resized to 64*64 dimension.

# Data augmentation
The Keras Image Generator is also used to augment training data by rotating
and shiftings transformations on the fly.

# Results
Using the model and accuracy of 82% was achieved.