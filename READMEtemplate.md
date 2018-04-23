# Behavioral Cloning Project

## Writeup report for project 3, term 1.

This is the report for project 2, term 1.

In this project a convolutional deep neural network model was used to effectively teach a car to drive autonomously in a simulated driving application to predict good driving behaviour from a human.

The total data used in this project is a union between *data provided by udacity* and *data collected by the student* using a [simulator from udacity team](https://github.com/udacity/self-driving-car-sim).


to apply deep learning principles to effectively teach a car to drive autonomously in a simulated driving application
 to predict good driving behaviour from human using a [simulator] gently provided by udacity team.

It was used [anaconda](https://www.anaconda.com/) Python flavour (version 3.6.1), [scikit-learn](http://scikit-learn.org) (version 0.19.1), [TensorFlow GPU](https://www.tensorflow.org/) (version 1.3.0), [keras](https://keras.io/) (version 2.1.5) *for fast modelling* and [OpenCV](https://opencv.org/releases.html) (version 3.4.0).

For the solution proposed, the next techniques were applied:

* [Digital Image scaling](http://graphics.csie.ncku.edu.tw/Image_Resizing/data/ImageResizing08.pdf)
* [Color space transformation](https://physics.info/color/).
* [Data cleansing](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&ved=0ahUKEwj4vJG9887aAhWPxFkKHQ32Ag4QFghLMAI&url=http%3A%2F%2Fwww.springer.com%2Fcda%2Fcontent%2Fdocument%2Fcda_downloaddocument%2F9780387244358-c2.pdf%3FSGWID%3D0-0-45-323094-p48087677&usg=AOvVaw3OdK2EQXKlUHXzO85eEyt6).
* [Data augmentation](https://openreview.net/forum?id=S1Auv-WRZ).
* [Data normalization](https://arxiv.org/pdf/1705.01809.pdf).
* [Shuffle training set](http://ieeexplore.ieee.org/document/8246726/?reload=true).
* [Batch Training](https://arxiv.org/abs/1711.00489).
* [Backpropagation](http://yann.lecun.com/exdb/publis/pdf/lecun-88.pdf).
* [Stochastic gradient based optimization](https://arxiv.org/abs/1412.6980).

This is a short list of keras funtions I used:
* [mean_square_error](https://keras.io/losses/). As Objective/Loss Function.
* [Adam](https://keras.io/optimizers/). Optimizer.
* [Sequential](https://keras.io/models/sequential/). For the model.
* [Lambda, Activation, Flatten, Dense, Dropout](https://keras.io/layers/core/). Layers used for the model.
* [Convolution2D](https://keras.io/layers/convolutional/). Convolutional Layers.
* [l2](https://keras.io/regularizers/). L2 regularization.

This is a non exclusive list of openCV functions I used:

* [cv2.resize()](https://docs.opencv.org/3.4.0/da/d6e/tutorial_py_geometric_transformations.html). Used for rescaling images to 32x32x3 size.
* [cv2.cvtColor()](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html).

*My solution to the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.*

**Note: This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo).**

---
## Reflection
To make it easier to follow up this reading the most relevant information is written on this readme file, however, you can find all step by step process with full explanation in detail and all coding lines on file *solution_behavioral_cloning.py*

Also the model performance videos can be viewed on the following links:
* model.h5 version0 (Zero) in track 1 (project).
* model.h5 version0 (Zero) in track 2 (challenge).
* model.h5 version1 (One) in track 1 (project).
* model.h5 version1 (One) in track 2 (challenge).

---
### Content of this repository
* A file named **model.py** with the code used to design the model of the solution proposed.
* A file named **drive.py** to be able to run the model on your local machine.
*  A file named **model.h5** with the trained model ready to test on the [simulator](https://github.com/udacity/self-driving-car-sim).

* A file named **report.md** with the report for the current project.

* The external links to check the videos for the project: [track1v0](https://youtu.be/QSoAm0VID_E), [track2v0](https://youtu.be/LIIXsI2CnoQ), [track1v1](https://youtu.be/HpTIpCGZP9k), [track2v1](https://youtu.be/pXws6qZlsi4).

* A folder named **model_and_videos_v0** with the short version of the videos for track1 and track2 with the behavioral for the version 0(zero) of the model designed.  [This is the link for full video *track1 model_v0*](https://youtu.be/QSoAm0VID_E) and [This is the link for full video *track2 model_v0*](https://youtu.be/LIIXsI2CnoQ).
*  A folder named **model_and_videos_v1** with the short version of the videos for track1 and track2 with the behavioral for the version 1(one) of the model designed.  [This is the link for full video *track1 model_v1*](https://youtu.be/HpTIpCGZP9k) and [This is the link for full video *track2 model_v1*](https://youtu.be/pXws6qZlsi4).


## Approach
For this project, steps provided were followed as advised in the [rubric](https://review.udacity.com/#!/rubrics/432/view) .

Disclaimer!: To solve this problem I was inspired on [this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for model design and data collection, [this post](https://medium.com/udacity/udacity-self-driving-car-nanodegree-project-3-behavioral-cloning-446461b7c7f9) for data preprocessing, on [this student approach for further data preprocessing](https://github.com/naokishibuya) and [this post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) to get shadowed images during the data augmentation process. All that said, thank you very much for reading this document!.

### Solution design process.

The solution aproach for this project is explained along this document.

Initially an image data set was collected usign the [simulator](https://github.com/udacity/self-driving-car-sim) to test some basic network architectures like a [Deep Fully Connected Network](https://arxiv.org/abs/1603.04930) and a [Convolutional Neural Network](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). 
Then further data was collected and unified to train and test the network performace. The data set was preprocessed (removed redundant data, data augmentation). Model architecture was selected and *parameters tunning process* was done by trial and error, testing the model performance on the [simulator](https://github.com/udacity/self-driving-car-sim).


#### Data collection.
The set of images used on this project was progresively constructed during the whole design aproach process. The paths for all images where gathered in a single csv file (*merged4_modified.csv*) with paths updated to run on a single linux server using python [pandas](https://pandas.pydata.org/) library and custom defined funtions  *mergecsv()* and *path_remove()* where defined (from code lines 40 to lines 80).

The final data set used to train and validate network performance contains as subset:
* Images provided by udacity team on the project.
* Images from one lap clock wise and one counterclockwise from track 1.
 * Some Images from track2. To help the model generalize.
* Also, during debugging process, a biased behavior to drive straigforward for the model was detected. So inspired on [this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) more images from left and rigth turns from track1 were extracted.

##### Training and validation data selection.
So basically, during the whole desing process more and more images where added to build up the training and validation data set. Ending with a total of 16235 images. 80 % of them used as initial training data set and the remaining 20% for validation. It is important to note that most of the images have had steering angles with values near 0(zero) as shown in mage below.

The initial data set was shuffled and then splited into training (80%) and validation(20%) data set.

**Note:** For training  set, during batch training, data was shuffle each time an image was feed into the model.

<br/> ![alt text][image1]
 
#### Data preprocessing (Part 1)
First, the general data set of 16235 images was [shuffled](https://www.tandfonline.com/doi/abs/10.1080/15536548.2012.10845652) (code line 90) and then splited in training(80%) and validation(20%) sets.  Data distributions are shown below.
<br/> ![alt text][image2]
<br/> ![alt text][image3]
 
It was noticed that most of the initial data (without preprocessing) used was linked to predicting responses to drive straighforward.
So *from code lines 138 to 320* data was preprocesed to get a more balanced distribution, close to a gaussian distribution.

Funtions get_list_pos() and find_pos() were defined to get the position of the path o the redundant images (code lines 138 to 193). Funtion my_hist(). 

Basically, data cleaning was done in two stages. 

On the first one, one threshold value of *1.5 times the average of the data* to remove reduntand information. That is, data images above the threshold_value(1.5*average)  were removed randomly until having each bin with a maximum of the therhold value. This is show in the image below.
<br/> ![alt text][image4]

On the seccond stage to get a better distribuion, closer to a guassian one, the same threshold value was defined (again,  threshold = 1.5*average).  For bins from the histogram with a sample number above the new threshold a probability to keep values was empirically defined as  **keepProb= (samples_at_current_bin/threshold)*0.5 ** (code lines 277 to 290), and keeping a close gaussian probability near the mean, redundanta data was removed. Resulting with a training set distribution as shown below.
<br/> ![alt text][image5]

So, before starting processing data a process of data cleaning was done, and initial data distribution(in blue) was filtered to get a better distributed training data (in orange) as shoen below.
<br/> ![alt text][image6]

####  Brief Solution description aproach.

Initially a simple deep fully connected network with 1 flatten layer and a dense  layer was trained to check if model was able to run in the simulator. 
Then inspired by [nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) I started with a simple convolutional neural network of 1 convolutional layer and progressively I was testing performance training the model while increasing the number of convolutional layers.

Then, noticing that network performance was still bad: I mean, that the car went out of the track, I decided to look for data augmentation and I started and seccond preprocessing part for training data and a third part doing data augmentation.

I tested out different models and parameters values but I just got two *fully working on track1*, but unfortunately,  both of them partially working on track 2.

So, I ended up with two models, one without regularization which is the one always mentioned on this document as **version Zero** and a seccond one with drop out and regularization, referred on this document as **version one**.

In summary,  I used a very similar model to the [nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)  including dropout layers and also adding L2 regularization technique to avoid overfitting. The final model description and parameters tunning is drescribed later on on this document.

#### Data preprocessing (Part 2)
Here custom functions and steps applied to images  are explained.


Images were loaded in RGB color space (function read_image() in line 327) . To improve generalization pf the model images where cropped (60 pixels) on top and (20 pixels ) at the bottom of each image (code line 342).
<br/> ![alt text][image7]
<br/> ![alt text][image9]



To reduce the number of trainable parameters images where resized to 64x64 pixels for each channel with function *resize_img()* (code lines 356 to 365).
<br/> ![alt text][image10]


As part of the experimentation process, images where converted to YUV color space as in [Nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) to check performance.
<br/> ![alt text][image8]

##### Data augmentation and model generalization
To augment the data a custom data_augmentation pipeline was define (function *preprocess_train_data()*), where is applied:
* random_image_select() : (Code lines 387-400) Used to select pseudo-randomly images from the center, left or rigth camera.
* random_flip(): ( Code lines 405 - 418). Used to flip images pseudorandomly.
* random_shadow(): (Code lines 423 - 457). Used to apply a random shadow mask on images.
* brigth_augmentation2(): (Code lines 468 - 479) used to shift sligthly the image brigth in HSV color space using a random_uniform deviation.
Some image samples are shown to ilustrate the preprocess pipeline:

<br/> ![alt text][image11]
<br/> ![alt text][image12]
<br/> ![alt text][image16]

#### Batch generators
Two custom funtion to generate bath for training and validation with the aim of avoiding overload of RAM memory where defined. The first one called *batch_generator2()*, used for debugging purposes, and, the seccond one called **batch_generator()**, used to produce **batch training** and **batch validation** data with a chosen batch of 64 images. 


#### Model Architecture 
As it was previously mentioned the final model architecture chosen is based on the [nvidia end-to-end learning model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with different number of output neurons, with addition of dropout layers all way along the network and with L2 regularization in the fully conected layers at the top. Also, I used "ELUs" as nonlinear layers because the given better results, on my model  calibration, than the "RELUs".

<br/> ![alt text][image20]

##### Data normalization
Training data was normalized before feeding it into the model using a  keras.layers.Lambda layer (code line 629).
##### Parameters Selection

The final parameters set up was the following:
* Lambda layer:
Applied  aproximation divinding by 255 and substractiong -0.5 to get pixel data between (-0.5) and (+0.5) on each channel.
* Convolutional layers:
As in the [nvidia model]((https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) ) I used strided convolutions in the first three convolutional layers with a 2x2, 5x5 kernels. Non strided convolutional layers with 3x3 kernels.
* Activation layers: 'ELU'
* Flatten layer:
Used to ensemble convolutional layers with fully connected layers.
* Dense layers:
Used as top abstraction layers with the last one with a single neuron in the output for regression task.

* Loss Funtion:
Mean squared error. Used for prediction/regression of steering angles.
* Learning rate: given the fact I used Adam optimizer starting with fixed learning rate it was adjusted by *keras.optimizers.Adam*.
* Optimization algortihm: 
Adam. Used because it adjustes the learning rate dirung the training process using stochastic gradient based optimization.
* batch size: 64. Use to work very fast on Nvidia GPU quadro P1000.
* Epochs: 30. Just enough to avoid overfitting.

So at the end we had:

Total params: 296,549
Trainable params: 296,549
Non-trainable params: 0

The last run accuraccy report can be view on file [p3t1accuracy_report.txt](https://github.com/Andrejkf/solution_behavioral_cloning_p3t1/blob/master/other_files/p3t1accuracy_report.txt "p3t1accuracy_report.txt").


## Results and discussion

With these starategies the model performed well (safety for pasengers) on track1 (test 1) but partially on track2 (challenge) track. 

Speed of the model was up to 15 dispacement/time units  and honestly I would like to have it working at maximum speed.

It was really hard to tune parameters for this model. Specifically because as the number of free parameters of the model increased, then the optimization complexity of the problem increased. (that means , I needed to start parameters tunning again.

Something noticeable is that introduction of L2 regularization in top layers (dense/fully connected layers) was very helpful for the model to generalize.

About dropout layers. They also helped to substantially increase the model abstraction capacity.


## To improve
I want to summarize the posible improvements that come to my mind for this submission at this point:

* Further data preprocessing. For example, by randomly shifting vertically and horizontally the training image data.

* More accurated data collection. To be honest, in most of mycollected data steering angle values were in most cases close to zero, so better data collection and non relevant data filtering should be added.

* To perform better on track 2 (challenge track),
collection of data images from seccond track in addition to the two previously mentioned  steps could be applied.

* A simple PID controller should be added by the author of this document in the file drive.py to make the model work better.




---
[//]: # (Image References)

[image1]: ./report_images/hist1.png "Initial data set"
[image2]:  ./report_images/hist2.png "Initial training set"

[image3]: ./report_images/hist3.png "initial validation set"

[image4]: ./report_images/hist4.png "training set redundant data removed"

[image5]:  ./report_images/hist5.png "training set filtered data"

[image6]:  ./report_images/hist6.png  "training set comparison"


[image7]:  ./report_images/preprocess1.png "sample image in rgb"

[image8]:  ./report_images/preprocess1_in_yuv.png "sample image in yuv"

[image9]:  ./report_images/preprocess2.png "cropped image"

[image10]:  ./report_images/preprocess3.png "resized 64x64 image"

[image11]:  ./report_images/preprocess4.png "flipped image"

[image12]:  ./report_images/preprocess5_img1.png "random shadow sample 1"

[image13]:  ./report_images/preprocess5_img2.png "random shadow sample 2"

[image14]:  ./report_images/preprocess5_img3.png "random shadow sample 3"

[image15]:  ./report_images/preprocess5_img4.png "random shadow sample 4"

[image16]:  ./report_images/preprocess6_img1.png "brightness augmentation sample 1"

[image17]:  ./report_images/preprocess6_img2.png "brightness augmentation sample 2"

[image18]:  ./report_images/preprocess7_img1.png "full preproced image sample 1"

[image19]:  ./report_images/preprocess7_img2.png "full preproced image sample 1"


[image20]: ./report_images/model_architecture.jpg "model architecture"



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MjI1MTEwMTQsMTMxMDA5NTgxNiwxMz
c3ODM4NzQxLDc1MjQ0ODA1LC0yMDQwNzkyMTUsNzQ4OTg3Njg2
LDM4MDIxMTkzMiwtMTI2NjczODI0NywxMTYwMjQzMTE2LC05NT
gwMzk5MzgsLTE2NTk2ODY5NjksLTEzMTc1MDAzMjgsMTQ5NzIz
MTI4MiwxNDM5MTI2NDA1LDExMTc3ODkzOTYsNjk2NTAwMDQsNz
k0MDQ3Njg0LC0yMDcyNTEyMDE0LDE2MTY1MDI5NzgsNTM0NzEy
NTk4XX0=
-->