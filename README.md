# MultiDigit_Captcha_CNN

This repository contains code written to predict the sum of the digits appearing in an image containing hand-written digits. 

### Input to the Model:
* 30000 images with labels (sum of the digits in the image)
* Used a train-val split of 80-20 resulting in 24000 training images and 6000 validation images

#### Methodology used:
* I tried multiple architectures, including but not limited to SimpleCNNs, Yann LeCun's LeNet-5 (Modified) for MNIST datasets, and the one described below

### Model
________________________________________________________________
Layer (type)                 Output Shape              Param  
________________________________________________________________
conv2d_12 (Conv2D)           (None, 38, 166, 32)       320       
_________________________________________________________________
batch_normalization_12 (Batc (None, 38, 166, 32)       128       
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 36, 164, 32)       9248      
_________________________________________________________________
batch_normalization_13 (Batc (None, 36, 164, 32)       128       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 18, 82, 32)        0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 16, 80, 64)        18496     
_________________________________________________________________
batch_normalization_14 (Batc (None, 16, 80, 64)        256       
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 14, 78, 64)        36928     
_________________________________________________________________
batch_normalization_15 (Batc (None, 14, 78, 64)        256       
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 7, 39, 64)         0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 5, 37, 128)        73856     
_________________________________________________________________
batch_normalization_16 (Batc (None, 5, 37, 128)        512       
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 3, 35, 128)        147584    
_________________________________________________________________
batch_normalization_17 (Batc (None, 3, 35, 128)        512       
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 1, 17, 128)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 2176)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 100)               217700    
_________________________________________________________________
dropout_4 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 74)                7474      
_________________________________________________________________
dropout_5 (Dropout)          (None, 74)                0         
_________________________________________________________________
dense_8 (Dense)              (None, 37)                2775      
_________________________________________________________________

### Total params: 516,173
### Trainable params: 515,277
### Non-trainable params: 896

________________________________
### Loss and Optimisation

* I used the "Categorical cross entropy loss" with "ADAM" optimizer using "ACCURACY" as the metric
* The best model was trained for 100 epochs with a batch size of 50

### Results

* The best model gave an accuracy of 93% on training data and a 80% accuracy on validation set
* The training loss was 0.3 and the validation loss was 0.9
* Some examples from the test dataset were plotted with correct and predicted labels and all of them were predicted correctly
* Confusion matrix was created
* Important metrics like Precision, Recall, F1 score along with their supports were calculated

### Future Work

* I plan to add another model to this dataset, which will in-principle do the hard-task of segmentation itself (using contour-detection, edge-detection or connected-component algorithms) and then use the padded, rescaled 4 numbers which have been segmented out and predict their labels. Then the sum of these labels would be the correct label. The model can be trained on MNIST dataset using SOTA architectures.
* This method would yield superior results as it does the hard task of segmentation itself, can have more data (MNIST alone is million images), can also use benchmark models for the MNIST data.

### References

* I read the following blogs/research papers/kaggle notebooks, and they have helped me finish the project

    #### Blogs

    * How to break a CAPTCHA system in 15 minutes with Machine Learning (https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710)
    * Solving CAPTCHAs — Machine learning vs online services (https://towardsdatascience.com/solving-captchas-machine-learning-vs-online-services-3596ad6f0137)
    * OCR model for reading Captchas (https://keras.io/examples/vision/captcha_ocr/)
    * Build a Multi Digit Detector with Keras and OpenCV (https://towardsdatascience.com/build-a-multi-digit-detector-with-keras-and-opencv-b97e3cd3b37)
    * How to Develop a CNN for MNIST Handwritten Digit Classification (https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)
    * Crack CAPTCHA using deep learning (https://mathematica.stackexchange.com/questions/143691/crack-captcha-using-deep-learning)

    #### Research Papers
    * Deep-CAPTCHA: a deep learning based CAPTCHA solver for vulnerability assessment
    * CAPTCHA Recognition with Active Deep Learning
    * Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks
    * MDig: Multi-digit Recognition using Convolutional Nerual Network on Mobile

    #### Github Repos
    * https://github.com/JackonYang/captcha-tensorflow
    * https://github.com/sambit9238/Deep-Learning

    #### Kaggle Notebooks
    * https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1
    * https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
    * https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist


# Running the Model

## On local
* git clone https://github.com/kushagragarwal2443/MultiDigit_Captcha_CNN.git
* pip3 install -r requirements.txt
* Open terminal and start the jupyter notebook
* Open view_data.ipynb
* Open train_data.ipynb

## On Google Colab
* Link to colab folder: https://colab.research.google.com/drive/1HEFDaln8sQwSL-NDQQIwMgTFjErkoS9_?usp=sharing

## On Ada
* Clone the repo
* sinteractive
* sbatch script.sh
* squeue -u kushagra2443
* cat ml4ns_mdcnn.txt to check progress

