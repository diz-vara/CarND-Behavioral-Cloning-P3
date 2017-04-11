# CarND P3 **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

 Changes in my drive.py file reflect operations performed in **generator** and include:
 *  image cropping (top 60 and bottom 20 rows removed from analysis)
 *  image resizing (downscale from 320x80 to 160x40)
 *  image normalization 




#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 8 and 64 (model.py lines 36-60) followed by two fully-connected ('Dense') layers with 100
and 50 units respectively. 

The model includes RELU layers to introduce nonlinearity on two first convolutional layers and PReLU (ReLU with lernable parameter of the negative slope) on other layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in fully-connected layers to reduce overfitting (model.py, lines 65 and 69). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with light weight decay (1e-7): it give slightly better results.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving from opposite directions and recovering from the  sides of the road ... 



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 36-70) consisted of a convolution neural network with 
4 convolutional hyper-layers consisting of convolution, batch normalization, ReLU (or PReLU) and
MaxPooling layers. In fierst hyper-layer, I replaced one 5x5 convolution with two consecutive 3x3 
convlutions.

the following layers and layer sizes ...

+--------------------+--------------------------+--------------+
| Layer (type)       |  Output Shape            | #params
+====================+==========================+==============+
| **First hyper-layer**
+--------------------+--------------------------+--------------+
| Convolution2D (3x3)| (None, 38, 158, 8)       |  224         |
+--------------------+--------------------------+--------------+
| BatchNormalization | (None, 38, 158, 8)       | 32           |
+--------------------+--------------------------+--------------+
| ReLU               | (None, 38, 158, 8)       |              |
+--------------------+--------------------------+--------------+
| Convolution2D (3x3)| (None, 36, 156, 16)      |  1168        |
+--------------------+--------------------------+--------------+
| BatchNormalization | (None, 36, 156, 16)      | 64           |
+--------------------+--------------------------+--------------+
| ReLU               | (None, 36, 156, 16)      | 0            |
+--------------------+--------------------------+--------------+
| MaxPooling2D       | (None, 18, 78, 16)       | 0            |
+--------------------+--------------------------+--------------+
| **Second hyper-layer**
+--------------------+--------------------------+--------------+
| Convolution2D (3x3)| (None, 16, 76, 24)       | 3480         |
+--------------------+--------------------------+--------------+
| BatchNormalization | (None, 16, 76, 24)       | 96           |
+--------------------+--------------------------+--------------+
| PReLU              | (None, 16, 76, 24)       | 29184        |
+--------------------+--------------------------+--------------+
| MaxPooling2D       | (None,  8, 38, 24)       | 0            |
+--------------------+--------------------------+--------------+
| **Third hyper-layer**
+--------------------+--------------------------+--------------+
| Convolution2D (3x3)| (None,  6, 36, 32)       | 6944         |
+--------------------+--------------------------+--------------+
| BatchNormalization | (None,  6, 36, 32)       | 128          |
+--------------------+--------------------------+--------------+
| PReLU              | (None,  6, 36, 32)       | 6912         |
+--------------------+--------------------------+--------------+
| MaxPooling2D       | (None,  3, 18, 32)       | 0            |
+--------------------+--------------------------+--------------+
| **Fourth hyper-layer**
+--------------------+--------------------------+--------------+
| Convolution2D (3x3)| (None,  1, 16, 64)       | 18496        |
+--------------------+--------------------------+--------------+
| BatchNormalization | (None,  1, 16, 64)       | 256          |
+--------------------+--------------------------+--------------+
| PReLU              | (None,  1, 16, 64)       | 1024         |
+--------------------+--------------------------+--------------+
| Flatten            | (None,  1024)            | 0            |
+--------------------+--------------------------+--------------+
+--------------------+--------------------------+--------------+
| **Dense layers**
+--------------------+--------------------------+--------------+
| Dense              | (None,  100)             | 102500       |
+--------------------+--------------------------+--------------+
| BatchNormalization | (None,  100)             | 400          |
+--------------------+--------------------------+--------------+
| Dropout            | (None,  100)             | 0            |
+--------------------+--------------------------+--------------+
| PReLU              | (None,  100)             | 100          |
+--------------------+--------------------------+--------------+
| Dense              | (None,  50)              | 5050         |
+--------------------+--------------------------+--------------+
| PReLU              | (None,  50)              | 50           |
+--------------------+--------------------------+--------------+
| Dropout            | (None,  50)              | 0            |
+--------------------+--------------------------+--------------+
| Dense              | (None,   1)              | 51           |
+--------------------+--------------------------+--------------+


Total params: 176,159
Trainable params: 175,671
Non-trainable params: 488



![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
