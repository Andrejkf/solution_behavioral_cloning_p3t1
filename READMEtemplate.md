# Behavioral Cloning Project
hello
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


So basically, during the whole desing process more and more images where added to build up the training and validation data set. Ending with a total of 16235 images. 80 % of them used as initial training data set and the remaining 20% for validation. It is important to note that most of the images have had steering angles with values near 0(zero) as shown in mage below.
<br/> ![alt text][image1]
 
#### Data preprocessing (Part 1)
First, the general data set of 16235 images was [shuffled](https://www.tandfonline.com/doi/abs/10.1080/15536548.2012.10845652) (code line 90) and then splited in training(80%) and validation(20%) sets. Data distributions are shown below.
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

To reduce the number of trainable parameters images where resized to 64x64 pixels for each channel with function *resize_img()* (code lines 356 to 365).

As part of the experimentation process, images where converted to YUV color space as in [Nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) to check performance.

##### Data augmentation and model generalization
To augment the data a custom data_augmentation pipeline was define (function *preprocess_train_data()*), where is applied:
* random_image_select() : (Code lines 387-400) Used to select pseudo-randomly images from the center, left or rigth camera.
* random_flip(): ( Code lines 405 - 418). Used to flip images pseudorandomly.
* random_sadow(): (Code lines 423 - 457). Used to apply a random shadow mask on images.
* brigth_augmentation2(): (Code lines 468 - 479) used to shift sligthly the image brigth in HSV color space using a random_uniform deviation.

#### Batch generators
Two custom funtion to generate bath for training and validation with the aim of avoiding overload of RAM memory where defined. The first one called *batch_generator2()*, used for debugging purposes, and, the seccond one called **batch_generator()**, used to produce **batch training** and **batch validation** data with a chosen batch of 64 images. 

#### Model Architecture 
As it was previously mentioned the final model architecture chosen is based on the [nvidia end-to-end learning model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with different number of output neurons, with addition of dropout layers all way along the network and with L2 regularization in the fully conected layers at the top. Also, I used "ELUs" as nonlinear layers because the given better results, on my model  calibration, than the "RELUs".








During the desing process, it was noted that the server machine took a long time for training data. initially in terms of days but with code improvements reduced to hours. The most advantaged part was to feed in t









Training and valdiation data selection.
Parameters  tunning:
Simulation.


### 1. Base Model and Adjustments

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (and later in the student forum, the [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py)) - the diagram below is a depiction of the nVidia model architecture.

<img src="./images/nVidia_model.png?raw=true" width="400px">

First I reproduced this model as depicted in the image - including image normalization using a Keras Lambda function, with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text - including converting from RGB to YUV color space, and 2x2 striding on the 5x5 convolutional layers. The paper does not mention any sort of activation function or means of mitigating overfitting, so I began with `tanh` activation functions on each fully-connected layer, and dropout (with a keep probability of 0.5) between the two sets of convolution layers and after the first fully-connected layer. The Adam optimizer was chosen with default parameters and the chosen loss function was mean squared error (MSE). The final layer (depicted as "output" in the diagram) is a fully-connected layer with a single neuron. [*`model.py` lines 268-329*]

### 2. Collecting Additional Driving Data

Udacity provides a dataset that can be used alone to produce a working model. However, students are encouraged (and let's admit, it's more fun) to collect our own. Particularly, Udacity encourages including "recovery" data while training. This means that data should be captured starting from the point of approaching the edge of the track (perhaps nearly missing a turn and almost driving off the track) and recording the process of steering the car back toward the center of the track to give the model a chance to learn recovery behavior. It's easy enough for experienced humans to drive the car reliably around the track, but if the model has never experienced being too close to the edge and then finds itself in just that situation it won't know how to react.

### 3. Loading and Preprocessing

In training mode, the simulator produces three images per frame while recording corresponding to left-, right-, and center-mounted cameras, each giving a different perspective of the track ahead. The simulator also produces a `csv` file which includes file paths for each of these images, along with the associated steering angle, throttle, brake, and speed for each frame. My algorithm loads the file paths for all three camera views for each frame, along with the angle (adjusted by +0.25 for the left frame and -0.25 for the right), into two numpy arrays `image_paths` and `angles`. [*`model.py` lines 174-211*]

Images produced by the simulator in training mode are 320x160, and therefore require preprocessing prior to being fed to the CNN because it expects input images to be size 200x66. To achieve this, the bottom 20 pixels and the top 35 pixels (although this number later changed) are cropped from the image and it is then resized to 200x66. A subtle Gaussian blur is also applied and the color space is converted from RGB to YUV. Because `drive.py` uses the same CNN model to predict steering angles in real time, it requires the same image preprocessing (**Note, however: using `cv2.imread`, as `model.py` does, reads images in BGR, while images received by `drive.py` from the simulator are RGB, and thus require different color space conversion**). All of this is accomplished by methods called `preprocess_image` in both `model.py` and `drive.py`. [*`model.py` lines 68-87*]

### 4. Jitter

To minimize the model's tendency to overfit to the conditions of the test track, images are "jittered" before being fed to the CNN. The jittering (implemented using the method `random_distort`) consists of a randomized brightness adjustment, a randomized shadow, and a randomized horizon shift. The shadow effect is simply a darkening of a random rectangular portion of the image, starting at either the left or right edge and spanning the height of the image. The horizon shift applies a perspective transform beginning at the horizon line (at roughly 2/5 of the height) and shifting it up or down randomly by up to 1/8 of the image height. The horizon shift is meant to mimic the hilly conditions of the challenge track. The effects of the jitter can be observed in the sample below. [*`model.py` lines 89-118*]

<img src="./images/sanity-check-take-4.gif?raw=true">

### 5. Data Visualization

An important step in producing data for the model, especially when preprocessing (and even more so when applying any sort of augmentation or jitter) the data, is to visualize it. This acts as a sort of sanity check to verify that the preprocessing is not fundamentally flawed. Flawed data will almost certainly act to confuse the model and result in unacceptable performance. For this reason, I included a method 'visualize_dataset', which accepts a numpy array of images `X`, a numpy array of floats `y` (steering angle labels), and an optional numpy array of floats `y_pred` (steering angle predictions from the model). This method calls `process_img_for_visualization` for each image and label in the arrays. [*`model.py` lines 57-66*]

The `process_img_for_visualization` method accepts an image input, float `angle`, float `pred_angle`, and integer `frame`, and it returns an annotated image ready for display. It is used by the `visualize_dataset` method to format an image prior to displaying. It converts the image colorspace from YUV back to the original BGR, applies text to the image representing the steering angle and frame number (within the batch to be visualized), and applies lines representing the steering angle and the model-predicted steering angle (if available) to the image. [*`model.py` lines 40-55*]

### 6. Data Distribution Flattening 

Because the test track includes long sections with very slight or no curvature, the data captured from it tends to be heavily skewed toward low and zero turning angles. This creates a problem for the neural network, which then becomes biased toward driving in a straight line and can become easily confused by sharp turns. The distribution of the input data can be observed below, the black line represents what would be a uniform distribution of the data points. 

<img src="./images/data_distribution_before_3.png?raw=true" width="400px">

To reduce the occurrence of low and zero angle data points, I first chose a number of bins (I decided upon 23) and produced a histogram of the turning angles using `numpy.histogram`. I also computed the average number of samples per bin (`avg_samples_per_bin` - what would be a uniform distribution) and plotted them together. Next, I determined a "keep probability" (`keep_prob`) for the samples belonging to each bin. That keep probability is 1.0 for bins that contain less than `avg_samples_per_bin`, and for other bins the keep probability is calculated to be the number of samples for that bin divided by `avg_samples_per_bin` (for example, if a bin contains twice the average number of data points its keep probability will be 0.5). Finally, I removed random data points from the data set with a frequency of `(1 - keep_prob)`.  [*`model.py` lines 215-248*]

The resulting data distribution can be seen in the chart below. The distribution is not uniform overall, but it is much closer to uniform for lower and zero turning angles.

<img src="./images/data_distribution_after.png?raw=true" width="400px">

*After implementing the above strategies, the resulting model performed very well - driving reliably around the test track multiple times. It also navigated the challenge track quite well, until it encountered an especially sharp turn. The following strategies were adopted primarily to improve the model enough to drive the length of the challenge track, although not all of the them contributed to that goal directly.*

### 7. Implementing a Python Generator in Keras

When working with datasets that have a large memory footprint (large quantities of image data, in particular) Keras python generators are a convenient way to load the dataset one batch at a time rather than loading it all at once. Although this was not a problem for my implementation, because the project rubric made mention of it I felt compelled to give it a try. 

The generator `generate_training_data` accepts as parameters a numpy array of strings `image_paths`, a numpy array of floats `angles`, an integer `batch_size` (default of 128), and a boolean `validation_flag` (default of `False`). Loading the numpy arrays `image_paths` (string) and `angles` (float) from the csv file, as well as adjusting the data distribution (see "Data Distribution Flattening," above) and splitting the data into training and test sets, is still done in the main program. 

`generate_training_data` shuffles `image_paths` and `angles`, and for each pair it reads the image referred to by the path using `cv2.imread`. It then calls `preprocess_image` and `random_distort` (if `validation_flag` is `False`) to preprocess and jitter the image. If the magnitude of the steering angle is greater than 0.33, another image is produced which is the mirror image of the original using `cv2.flip` and the angle is inverted - this helps to reduce bias toward low and zero turning angles, as well as balance out the instance of higher angles in each direction so neither left nor right turning angles become overrepresented. Each of the produced images and corresponding angles is added to a list and when the lengths of the lists reach `batch_size` the lists are converted to numpy arrays and yielded to the calling generator from the model. Finally, the lists are reset to allow another batch to be built and `image_paths` and `angles` are again shuffled.  [*`model.py` lines 120-149*]

`generate_training_data` runs continuously, returning batches of image data to the model as it makes requests, but it's important to view the data that is being fed to the model, as mentioned above in "Data Visualization." That's the purpose of the method `generate_training_data_for_visualization`, which returns a smaller batch of data to the main program for display. (*This turned out to be critical, at one point revealing a bug in my implementation of `cv2.flip` causing the image to be flipped vertically instead of horizontally*)  [*`model.py` lines 152-168*]

### 8. More Aggressive Cropping

Inspired by [David Ventimiglia's post](http://davidaventimiglia.com/carnd_behavioral_cloning_part1.html?fb_comment_id=1429370707086975_1432730663417646&comment_id=1432702413420471&reply_comment_id=1432730663417646#f2752653e047148) (particularly where he says "For instance, if you have a neural network with no memory or anticipatory functions, you might downplay the importance of features within your data that contain information about the future as opposed to features that contain information about the present."), I began exploring more aggressive cropping during the image preprocessing stage. This also required changes to the convolutional layers in the model, resulting in a considerably smaller model footprint with far fewer parameters. Unfortunately, I was not successful implementing this approach (although the reason may have been because of an error in the `drive.py` image preprocessing), and ultimately returned to the original nVidia model and my original preprocessing scheme.

### 9. Cleaning the dataset

Another mostly unsuccessful attempt to improve the model's performance was inspired by [David Brailovsky's post](https://medium.freecodecamp.com/recognizing-traffic-lights-with-deep-learning-23dae23287cc#.linb6gh1d) describing his competition-winning model for identifying traffic signals. In it, he discovered that the model performed especially poorly on certain data points, and then found those data points to be mislabeled in several cases. I created `clean.py` which leverages parts of both `model.py` and `drive.py` to display frames from the dataset on which the model performs the worst. The intent was to manually adjust the steering angles for the mislabeled frames, but this approach was tedious, and often the problem was with the model's prediction and not the label or the ideal ground truth lay somewhere between the two. A sample of the visualization (including ground truth steering angles in green and predicted steering angles in red) is shown below.

<img src="./images/sanity-check-take-5.gif?raw=true">

### 10. Further Model Adjustments

Some other strategies implemented to combat overfitting and otherwise attempt to get the car to drive more smoothly are (these were implemented mostly due to consensus from the nanodegree community, and not necessarily all at once):

- Removing dropout layers and adding L2 regularization (`lambda` of 0.001) to all model layers - convolutional and fully-connected
- Removing `tanh` activations on fully-connected layers and adding `ELU` activations to all model layers - convolutional and fully-connected
- Adjust learning rate of Adam optimizer to 0.0001 (rather than the default of 0.001)

These strategies did, indeed, result in less bouncing back and forth between the sides of the road, particularly on the test track where the model was most likely to overfit to the recovery data.

### 11. Model Checkpoints

One useful tool built into the Keras framework is the ability to use callbacks to perform tasks along the way through the model training process. I chose to implement checkpoints to save the model weights at the end of each epoch. In a more typical application of neural networks, it might make more sense to simply end the learning process once the loss stops improving from one epoch to the next. However, in this application the loss was not an entirely reliable indicator of model performance, so saving model weights at the end of each epoch is something of a buy-one-get-X-free each time the training process runs. In the end, it was the weights from the third epoch of training that performed best and completed both test and challenge tracks.

### 12. Further Data Distribution Flattening

At one point, I had decided I might be throwing out too much of my data trying to achieve a more uniform distribution. So instead of discarding data points until the distribution for a bin reaches the would-be average for all bins, I made the target *twice* the would-be average for all bins. The resulting distribution can be seen in the chart below. This resulted in a noticeable bias toward driving straight (i.e. problems with sharp turns), particularly on the challenge track. 

<img src="./images/data_distribution_after_3.png?raw=true" width="400px">

The consensus from the nanodegree community was that underperforming on the challenge track most likely meant that there was not a high enough frequency of higher steering angle data points in the set. I once again adjusted the flattening algorithm, setting the target maximum count for each bin to *half* of the would-be average for all bins. The histogram depicting the results of this adjustment can be seen in the chart below. (*Note: the counts for the bins differ from the chart above because the dataset for the chart below includes both Udacity's and my own data.*)

<img src="./images/data_distribution_after_4.png?raw=true" width="400px">

## Results 

These strategies resulted in a model that performed well on both test and challenge tracks. The final dataset was a combination of Udacity's and my own, and included a total of 59,664 data points. From these, only 17,350 remained after distribution flattening, and this set was further split into a training set of 16,482 (95%) data points and a test set of 868 (5%) data points. The validation data for the model is pulled from the training set, but doesn't undergo any jitter. The model architecture is described in the paragraphs above, but reiterated in the image below:

<img src="./images/model_diagram.jpeg?raw=true" width="400px">

## Conclusion and Discussion

This project - along with most every other exercise in machine learning, it would seem - very much reiterated that it really is *all about the data*. Making changes to the model rarely seemed to have quite the impact that a change to the fundamental makeup of the training data typically had. 

I could easily spend hours upon hours tuning the data and model to perform optimally on both tracks, but to manage my time effectively I chose to conclude my efforts as soon as the model performed satisfactorily on both tracks. I fully plan to revisit this project when time permits.

One way that I would like to improve my implementation is related to the distribution flattening scheme. As it is currently implemented, a very large chunk of data is thrown out, never to be seen again. I find this bothersome, and I feel that wasting data like this (even if it is mostly zero/near-zero steering angles) is a missed opportunity to train a model that can better generalize. In the future, I would like to pass the full dataset (`image_paths` and `angles`) to the generator and allow it to flatten the distribution itself, throwing out a different portion of the data each epoch or batch.

I would also like to revisit implementing a more agressive crop to the images before feeding them to the neural net. Another nanodegree student achieved good model performance in a neural net with only 63 parameters. Such a small neural net is desirable because it greatly reduces training time and can produce near real-time predictions, which is very important for a real-world self-driving car application.

I enjoyed this project thoroughly and I'm very pleased with the results. Training the car to drive itself, with relatively little effort and virtually no explicit instruction, was extremely rewarding.

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

[image20]:  ""



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0OTI5NTQ5MiwxNDk3MjMxMjgyLDE0Mz
kxMjY0MDUsMTExNzc4OTM5Niw2OTY1MDAwNCw3OTQwNDc2ODQs
LTIwNzI1MTIwMTQsMTYxNjUwMjk3OCw1MzQ3MTI1OTgsLTk5MT
kyNTUzNCwtOTA5OTc1MzA2LC02ODc4NTkzNiwxNDU2NTIxNzA1
LDE2MDQ0ODQ4NjUsMjAwNzg1NDA5MSwxNTM0NTAyNDQ4LDIzMj
MxNTA0NiwtMTgzMTc1MzAyNCwxOTQzODk5MzkwLDE3MDIzODM2
OTVdfQ==
-->