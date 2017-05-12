#**Behavioral Cloning** 

Raymond Leung
2017-05-10

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior  -- COMPLETED /emdata/
* Build, a convolution neural network in Keras that predicts steering angles from images -- COMLETED model.py
* Train and validate the model with a training and validation set -- COMPLETED model.h5
* Test that the model successfully drives around track one without leaving the road  -- COMPLETED video.mp4
* Summarize the results with a written report -- COMPLETED writeup_template.md 


[//]: # (Image References)

[image1]: ./figure_1.png "Model Mean Squared Error Loss"
[image2]: ./trainrun.mp4 "Training Run"
[image3]: ./video.mp4 "Autonomous Drive Complete Rn"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model  -- SUBMITTED 
* drive.py for driving the car in autonomous mode  -- SUBMITTED, NO MODIFICATION	
* model.h5 containing a trained convolution neural network -- SUBMITTED
* writeup_report.md or writeup_report.pdf summarizing the results -- SUBMITTED

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Basically I am using the Nvidia Architecture with slight change by adding a dropout layer and cropping the unnessary part

------------------------------------------------------------
| Normalized			| Lamda x/255 - 0.5                |
| Cropping images       | Cropping top 70, bottome 25      |
| Convolution 2D	    | 24 5x5 stride 2x2 and relu       |
| Convolution 2D	    | 36 5x5 tride 2x5 and relu        |
| Convolution 2D	    | 48 5x5 stride 2x2 and relu       |
| Convolution 2D	    | 48 5x5 stride 2x2 and relu       |
| Convolution 2D	    | 48 5x5 stride 2x2 and relu       |
| Flatten			    |       						   |
| Dropout			    | 0.5      						   |
| Dense 			    | Output 100      		     	   |
| Dense 			    | Output 50      		     	   |
| Dense 			    | Output 10      		     	   |
| Dense 			    | Output 1      		     	   |
------------------------------------------------------------




#### 2. Attempts to reduce overfitting in the model

Add a Dropout layer with prob = 0.5 after the Flatten to provide overfitting.   Without the Dropout layer, the car still able to finish the lap in this model, but it aligned
to the right side. It is improved after adding a drop out layer.


#### 3. Model parameter tuning

The adam optimizer parameter is default learning rate = 0.001

#### 4. Appropriate training data

I combined four set of training data
* completed lap on counter clockwise
* completed lap on clockwise
* start from the almost offtrack near the mud land.  Aggretive to turn left back to center  (repeated 2 times)
* start from before almost offtrack near the mud land.   Slightly turn left back to cetnter (repeated 2 times)



### Model Architecture and Training Strategy

#### 1. Solution Design Approach


THe first approach is using default data and LeNet model.   Not really working, the error rate is too high and get off track almost immediate after start.

Then added augmented images by flipping the center images, changed the LeNet model to Nvidia model.  Much better the error rate goes down significantly, able to pass the bridge.

Add left and right camera image with suggest correction steering value = 0.2.   Able to pass the bridge but keep slide out to the mud land after the bridge.  

Start using my own training data, first with 1.5 counter clockwise lap, then add a reverse lap.   Seems more stable but still have same problem.   Add a couple recovery around the offtrack area, plus recovery track before the offtrack area.   It improved but still failed some times.

Increase the correction value to 0.25, seems better.   Tried couple round with higher value and seems best result with correction steering = 0.3.   It is able to correct my off track ifself with this value.

The car is more aligned to the right hand side of the road, probably get overfitting the data on that.  Add a 0.5 probability Dropout.  The car is moving back to the center lane.  However, from some wide turn, it stay on left side before it do the big turn.   I think it is from training data.   But no offtrack after 10 laps.


#### 2. Final Model Architecture
------------------------------------------------------------
| Normalized			| Lamda x/255 - 0.5                |

| Cropping images       | Cropping top 70, bottome 25      |

| Convolution 2D	    | 24 5x5 stride 2x2 and relu       |

| Convolution 2D	    | 36 5x5 tride 2x5 and relu        |

| Convolution 2D	    | 48 5x5 stride 2x2 and relu       |

| Convolution 2D	    | 48 5x5 stride 2x2 and relu       |

| Convolution 2D	    | 48 5x5 stride 2x2 and relu       |

| Flatten			    |       						   |

| Dropout			    | 0.5      						   |

| Dense 			    | Output 100      		     	   |

| Dense 			    | Output 50      		     	   |

| Dense 			    | Output 10      		     	   |

| Dense 			    | Output 1      		     	   |
------------------------------------------------------------

Training Mean Square Loss[image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

Training Video[image2]


##### 4. Final Result
The video show the car is able to drive at least 2 laps autonomously. 

Autonomous Video[image3]
