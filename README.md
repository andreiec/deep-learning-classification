# Generated Images Classification

## Introduction
  ‘Generated Images Classification’ was a contest hosted on Kaggle during the Deep Learning course within the Faculty of Mathematics and Computer Science, University of Bucharest. The main goal for students was to find the most efficient methods for classifying images into 100 different categories, labeled from 0 to 99. The images were in color (RGB) and had a size of 64x64 pixels. Model training was conducted on the training set, which comprised 13000 images, validation on a set of 2000 images, and testing on 5000 images.

## Image Classification
  The image classification process covers a very wide sphere of the field called 'Artificial Intelligence', representing one of the most important concepts shaping the world in which we live. For this reason, we must be careful about the methods we try to implement in this area.

## Image data
  The images available for this contest are RGB type and 64x64px in size. Image reading and preprocessing. The reading of the images was done using the 'skimage' module so that it was done as quickly and efficiently as possible. In order to normalize the data, I used the ‘StandardScaler’ class from ‘sklearn’

## Aproached methods
  The methods that I used to try to get a bigger score are a modified Residual Network (ResNet10) and a custom Convolutional Neural Network (CNN).

### Residual Network
  Residual Network or, shortly, ResNet, was developed to address the challenges associated with training very deep neural networks. It uses skip links or shortcuts that enable the network to bypass one or more levels during training. By reducing the impact of the vanishing gradient issue, these skip connections facilitate the training of deeper networks. The model is pretty robust, having around 2900000 params. The training time on this model was around 14 hours for 30 epochs and only reached a validation accuracy of ~55%.

### Convolutional Neural Network
  Convolutional Neural Network is a neural algorithm within the concept of Deep Learning that has as input an image on which it calculates various patterns that it considers 'important', thus, the problem of classification representing only the highlighting of these patterns as much as possible. Training time was 1 hour and 30 minutes for 100 epochs and the validation accuracy was ~67%.

- Conv2D Layer

  These layers have as their main purpose the detection of patterns made after the application of several filters. There are a total of 4 convolutional layers with the following number of filters - 64, 64, 128, 256, each one having a kernel size of 3.

- Filters

  The main purpose of the filters is to detect patterns by calculating (together with the kernel) several values obtained after applying the values within the filters. These filters can be analogous to edge detection, emboss or bevel.

- Kernel

  The kernel is nothing more than a 'window' in which filter values are calculated. This kernel walks through each pixel and contains the values from the filters, and after the calculations, the pixel is assigned the value of the mean of the filter. A 3x3 kernel includes a window size of 3 pixels by 3 pixels.


- Input Shape

  Since the images are somehow small, we considered that the color of the image can be a very important feature (usually, image classification is performed on grayscale images). Therefore, I took all 3 color channels, resulting in an input shape of 64x64x3.


- Activation function

  For the current model, we used the most widely used activation function in convolutional neural networks - Relu.

- Dense layer

  The second important part of the chosen model is represented by the three Dense layers. These layers are the most used when it comes to neural networks as they fulfill their main purpose as efficiently as possible. The dense layer has as input the scalar product of all neurons and weights from the previous layer, and as output the sum of the values passed through the activation function.

  The first layers have 256 neurons and the last 100. The third dense layer has only 100 neurons because this is the output layer of the network for the 100 classes on which we have to make classifications. The first two layers have the activation function 'relu', while the last one is 'softmax' because we are left, after all the equations made, with a list of probabilities, and the classifier will choose the value with the highest probability as the identified class.

- Dropout layer

	Convolutional neural networks trained on small data sets can often experience the concept of overfitting on the training data. To solve this problem, the dropout layer temporarily removes the connections between various neurons to try to see the image 'differently'. The parameter in parentheses is the percentage of links it drops.

- Batch Normalization layer
  The batch normalization layer tries to resolve the internal covariate shift issue. The term "internal covariate shift" describes how, layer by layer, the model trains and the distribution of network activations changes. This may increase the difficulty of the training and impede convergence.

## Hyperparameter tuning

  Hyperparameter tuning is the process of determining the ideal collection of hyperparameters for a machine learning model. Hyperparameters refer to the external configuration settings of a model that are not acquired through training from the data. Rather, they are predetermined and have an impact on the training procedure itself.

  For the custom convolutional network I used a manual approach for searching the best parameters for the optimizer, the starting learning rate and the number of epochs. As it shows from the table, the best values were ‘Adam’ as optimizer, 0.001 as starting learning rate and 100 for number of epochs. Those values were used for training the final model. The whole process took more than an entire day of continuous training and processing.

## Conclusions

Even if there is a lot of room for improvements and tweaking, the models shown both score more than 50%, which is good for a dataset consisting of 100 classes. The custom convolutional network scored a 67% in the private leaderboard. As of future work, more model architectures and techniques could be used.


