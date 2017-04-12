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

[model]: ./model.png "Model Visualization"
[recov0]: ./examples/recov_0.jpg "Recovery Image"
[recov1]: ./examples/recov_10.jpg "Recovery Image"
[recov2]: ./examples/recov_20.jpg "Recovery Image"
[recov3]: ./examples/recov_30.jpg "Recovery Image"
[normal]: ./examples/ex_0_orig.jpg "Normal Image"
[flipped]: ./examples/ex_0_mirror.jpg "Flipped Image"
[cropped]: ./examples/ex_0_crop.png "Flipped Image"

[track1]: ./track1.mp4 "Flipped Image"

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


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, driving from opposite directions and recovering from the  sides of the road.
My appoach to training data creation I describe further.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I've decided to start with a simple and straighforward model. So, I constructed the convolution
network consisting of sequence of 3x3 convoulutions with PReLU activations, BatchNormalizations
and MaxPooling layers with moderate increase of number of features.
Originally, I planned to play with more sofisticated models at the end, but it appeared
that this model works, and I only simplified it a little bit (removed one layer after
reducing the image size and replaced two first PReLU activations by simple ReLUs to
decrease the number of trained parameters).


#### 2. Final Model Architecture

The final model architecture (model.py lines 36-70) consisted of a convolution neural network with 
4 convolutional hyper-layers consisting of convolution, batch normalization, ReLU (or PReLU) and
MaxPooling layers. In fierst hyper-layer, I replaced one 5x5 convolution with two consecutive 3x3 
convlutions.


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

Here is the graphical represenation of the model (it took a huge amount of time to create, but the solution was simple: replace obsolete
pydot2 with pydot3)

![alt text][model]

The model used an adam optimizer, with light weight decay (1e-7): it give slightly better results.

Model were trained for 7 epochs, as validation loss usually stopped to decrease afther this moment.


#### 3. Creation of the Training Set & Training Process

Dataset creation was the most time-consuming part of the project (but it is true for all real projects: to get good results, you 
need **a lot** of data).

In my data collection, I followed instructions from the lesson: first, I recorded two laps on track one using center lane driving. Here is an example: 

![alt text][normal]

Then I added one lap in the opposite (clock-wize) direction - and 'augmented' the data by flipping the image (and inverting the value of the angle).
Here is an example of image flipping:

![alt text][normal]
![alt text][flipped]

Being tested in the autonomouse mode, car reached the bridge - but left the track on the first turn after.

Then I added  several recordings of the 'smooth' passing of difficult corneds - and recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn
how to return to the center in case it goes slightly off.
Here are trhee frames from 'recovery' track.

![alt text][recov0]
![alt text][recov2]
![alt text][recov3]

In addition, I added images from left and right cameras to the analysis, adjusting steering angle by the value of 0.15.

'Autonomous' car behaved better - but didn't reach the finish.

Then I cropped the image (cutting off 60 lines from the top and 20 lines from the bottom):

![alt text][cropped] 
-- and car **finished the first track!**

[track1] (https://github.com/diz-vara/CarND-P3/blob/master/track1.mp4) 

At that point, data consisted of 13677 original samples, with additional cameras and mirroring it gave total of  82062 images, 65450 (80%) of them
were used for training, and 16412 (20%) - for validation.

When I tried to use trained model on the second track, the result was... No good, as you can imaging. But when, after several attempts, I've managed
to pass this track (and record two laps) - it appeared that model, trained on both tracks, can drive both of them! - On maximal speed (30 mph) on 
track1, and at the speed up to 10 mph on the second track.

At that point dataset consisted of 20464 'waypoints' (122784 images after augmentation);

Then I simplified things a little bit: I scaled the image from original 160x320 to 80x160 - and removed one 'hyper-layer' from the network.
It reduced the size of the model - but did not affect the result: it still succeded on both tracks.

Up to this point, I didn't 'generator' tecnique: due to the small size of the image, entire dataset
fitted in TitanX memory. But, as in real life I deal with a large images, I decided to take a try.

In 'generator' model, image cropping, resizing and normalization were moved from the model to the
generato function - and the same changes were made in **drive.py** file. 

New model worked on track1 not so good (car swayed on straight parts of the track) - but still
passable. But it couldn't drive on track2, going off the road on one or another (depending on
speed and initial conditions) point.
I've added more track2 recordings (one more circle + two laps in opposite directions) - but it did
not improve the situation.

What is the matter?

I think, the answer is very simple.
Without 'generator', I generated augmenterd data first, and it consisted of (N * 6) samples.
When I took 80% of these into the training data, the probability of any original point (or one of its
clones) to be presented in the training set, is very high.

On the other point, in 'generator' model I 'cut off' 20% of the data **before** augmentation,
and the model really never see it.


Finally, I want to say it would be nice to have one additional 'test' track - as simple as the 
first one, but not accessible for training. The it would be a chanse to see if our model really 
learns to *generalize* - or it just tryes to memorize all it see.

  
 


