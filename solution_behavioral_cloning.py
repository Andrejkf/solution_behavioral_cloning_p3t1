#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:55:23 2018

@author: andrej
"""
# cd /home/andrej/ownCloud/CarND-Behavioral-Cloning-P3-master-v4
# Import all libraries used in hte project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import  Lambda, Activation, Flatten, Dense, Dropout #, , , , , Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

# define a random seed for pseudo random numbers generation.
np.random.seed(0)

# Define paths
file_path_in =  './sim_data_15/merged4.csv'   
file_path_in_modified = './sim_data_15/merged4_modified.csv'

path_in = './sim_data_15/IMG/'
path_dir= './sim_data_15/IMG/'



# 1. Data collection
# 1.1. Merge all images and csv files in one songle folder.
def mergecsv():
    '''
    Function to merge all csv files with images information.
    f1,f2,f3: (.csv files) csvfiles that will be merged in a single one.
    '''
    f1 = pd.read_csv('./sim_data_13/merged2.csv', header= None)
    f2 = pd.read_csv('./simdata_10v2/driving_log.csv', header= None)
    f3 = pd.read_csv('./simdata_10v3/driving_log.csv', header= None)
    
    merged4 = pd.concat([f1,f2[1:],f3[1:]])
    merged4 = pd.concat([f1,f2[1:]])
    pd.DataFrame.to_csv( merged4, './sim_data_15/merged4.csv', index=None, header=None)



# 1.2.1 Handy funtion to remove original path from images list and just keep name of image files.
def path_remove(pathInFile, pathOutFile):
    '''
    Removes original path for image list and creates a new csv file with just the image_names on it.
    pathInFile: full input path + input_file_name
    pathOutFile: full output path + output_file_name
    '''
    import pandas as pd
    import csv
    lines = []
    with open (pathInFile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line2 = []
            line2.append(line[0].split('/')[-1])
            line2.append(line[1].split('/')[-1])
            line2.append(line[2].split('/')[-1])
            line2.extend(line[3:])
            lines.append(line2)

    pd_lines = pd.DataFrame.from_records(lines)
    pd.DataFrame.to_csv( pd_lines, pathOutFile, index=None, header=None)
    print('new file create at: ' , pathOutFile)
    
# 1.2.2. Remove original iamge path and update to local machine.
path_remove(file_path_in, file_path_in_modified)
    
# 2. Data preprocessing.

# 2.1. Import data(images names, steering angle, throttle, brake and speed values) using pandas.

var_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data_list = pd.read_csv('./sim_data_15/merged4_modified.csv', skiprows=[0], names=var_names, converters={'left': str.lstrip, 'right': str.lstrip})

# 2.2. Get the full list of images and asociated values.
data_list2 = shuffle(data_list)

# 2.3. Split data in training and validation set.
data_train, data_valid = train_test_split(data_list2, test_size = 0.2)

# Training set
centerImg   = data_train.center.tolist()
leftImg     = data_train.left.tolist()
rigthImg    = data_train.right.tolist() 
steering    = data_train.steering.tolist()
throttle    = data_train.throttle.tolist()
brake       = data_train.brake.tolist()
speed       = data_train.speed.tolist()

# Validation set
centerImgV   = data_valid.center.tolist()
leftImgV     = data_valid.left.tolist()
rigthImgV    = data_valid.right.tolist() 
steeringV    = data_valid.steering.tolist()
throttleV    = data_valid.throttle.tolist()
brakeV       = data_valid.brake.tolist()
speedV       = data_valid.speed.tolist()


# Rename validation_input and validation_labels.
X_valid   = centerImgV
y_valid   = np.float64(steeringV)

# 2.4 Plot histogram to check data distribution
plt.figure()
plt.hist(data_list2.steering)
plt.xlabel('Steering values')
plt.ylabel('Samples')
plt.title('Full initial data set distribution')

plt.figure()
plt.hist(steering)
plt.xlabel('Steering values')
plt.ylabel('Samples')
plt.title('Initial training set Steering values')

plt.figure()
plt.hist(steeringV)
plt.xlabel('Steering values')
plt.ylabel('Samples')
plt.title('Validaton set Steering values')


# 2.5. BALANCE DATA ( AT LEAST GET A LESS UNBALANCED DATA SET)
# Get a less umbalanced training data set to avoid over fitting or model predisposition in predictions

# 2.5.1 Define custom functions to get images based on the steering angle values.

# 2.5.1.1.  Function used to get image position row(on a variable with the list of all images names and steering values) for a given steering angle value.
def get_list_pos(steering, bins, delta_hist):
    '''
    Note: you should calculate the histogram using numpy.hist before using this function.
    Note 2: this function used function "find_pos". Custom funtion defined by the author.
    
    Returns a list of list with all positions of the values for each bin of the 1D- histogram.
    ( in this case : delta_hist>0. That is. get bins where steering values are above a given threshold.)
    
    steering: (list) vector with steering angle values. (Note: each value of the steering angle you enter should be of type "float").
    bins: bins of the histogram (that should be done previously).
    delta_hist: vector with the number of values above certain threshold that we want to eliminate to avoid the bias of the car to drive straigth.
    '''
    import numpy as np
    
    # define condition to filter values from bins : in this case : delta_hist>0. That is. get bins where valeus are above a given threshold.
    to_remove= np.where(delta_hist>0)[0]
    
    # get the list of list (with all positions of the sterring values values for each bin)
    positions= []
    for i in to_remove:
        positions.append( find_pos(steering, bins[i], bins[i+1]) )
    
    return positions

# 2.5.2 Function used to get a list of positions of images for a given steering angle
def find_pos(vector, position_left, position_rigth):
    '''
    Returns (a list with) the positions of values contained in a segment(subset of) the given input "vector", where:   left_boundary <= ( some values from original vector should be here) < rigth_boundary
    
    vector : (list) input vector with steering values.
    position_left : (float or int) left boundary of the segment.
    position_rigth: (float or int) rigth boundary of the segment.
    
    '''
    import numpy as np
    # define lower and upper limits for the segment
    left_limit  = position_left
    rigth_limit = position_rigth
    
    # get positions of the ( values greater or equal than left boundary of the segment )
    a = np.array(np.where(vector>=left_limit ), dtype= np.int )[0]  # values condition left
    # get positions of the ( values lower than the rigth boundary of the segment )
    
    b= []
    for i in a:
        # condition to get positions of the (values lower than rigth boundary)
        if vector[i] < (rigth_limit):
            b.append(i)
    # finally: b is a list with the positions of the values that satisfies the condition : left_limit < values < rigth_limit
    return b


# Funtion defined to understand how to filter data based on histograms (not used on the project, I just boult it to deduce how the function np.histogram works)
def my_hist(steering, nbins):
    '''
    '''
    import numpy as np
    
    counter= np.zeros(20)
    a = np.linspace(np.min(steering), np.max(steering), nbins +1)
    for i in range(len(steering)):
        for j in range(nbins-1):
            if steering[i] >= a[j] and steering[i] < a[j+1]:
                counter[j]+= 1
        if steering[i] >= a[nbins-1] and steering[i] <= a[nbins]:
            counter[nbins-1]+=1
    
    return counter, a


# 2.5.3. (Preprocessing Part 1) REMOVE REDUNDANT DATA
# All histograms where done with 20 bins.
# Ideally you should have the same amount of data for each steering angle value.
# so to have a more plain distribution or at least a near gaussian data distribution:
# A threshold value is defined and data above the thershold is eliminated.
# Note: The eliminated data is the most common data , to have more diversity in the data set.


# number of bins
nbins = 20
#get histogram
hist, bins = np.histogram(steering, bins= nbins)

#compute average value in histogram
avg = len(steering)/nbins
# define a htreshold value
threshold = int(avg*1.5)


# get vector with values above given threshold
delta_hist = hist -threshold

# get list of list of positions with values above given threshold
pos_over_threshold = get_list_pos(steering, bins, delta_hist)

# for this case we are filtering just values beween interval (bins[10], bins[11])
pos_over_threshold = shuffle( pos_over_threshold[0])

# get the position of the histogram values corresponding to the bins interval, that in this case corresponds to just values beween interval (bins[10], bins[11])
hist_pos_over_threshold = np.where(delta_hist>0)[0]
for j in hist_pos_over_threshold:
    # randomly remove steering values. As much as delta_hist_vector says for the corresponding bins.
    assert ( len(pos_over_threshold) > delta_hist[j] )
    select_list = []
    for i in range(delta_hist[j]):
        selected = np.random.randint(len(pos_over_threshold))
        select_list.append(pos_over_threshold[selected])
        pos_over_threshold.remove(pos_over_threshold[selected])
        
        
# This vector contains the positions of the images names to reduce from the training data set.
removeList = select_list

centerImg  = np.delete(centerImg, removeList, axis=0)
leftImg    = np.delete(leftImg, removeList, axis=0)
rigthImg   = np.delete(rigthImg, removeList, axis=0)
steering   = np.delete(steering, removeList, axis=0)


# 2.5.4. (Preproccessing  part 2) GET A LESS UNBALANCED DATA

#
nbins = 20

hist, bins = np.histogram(steering, bins= nbins)

# Plot data afte removing reduntant data (from previous step)
plt.figure()
plt.hist(steering, bins=20)
plt.xlabel('Steering values')
plt.ylabel('Samples')
plt.title('Training set removing redundant data')

avg = len(steering)/nbins
threshold = avg*1.5

# 2.5.4.1 Define probability to keep images for some steering angles
keepProb = []
for i in range(nbins):
    if hist[i] > threshold:
        keepProb.append(1/(hist[i]/threshold) * 0.5 )
    else:
        keepProb.append(1)



plt.bar(bins[:-1],hist/threshold,0.05)

removeList = []
for i in range(len(steering)):
    for j in range(nbins):
        if steering[i] >= bins[j] and steering[i] < bins[j+1]:
            if np.random.rand() > keepProb[j]:
                removeList.append(i)


# 2.5.4.2. Delete images to get a better data training distribution
centerImgF  = np.delete(centerImg, removeList, axis=0)
leftImgF    = np.delete(leftImg, removeList, axis=0)
rigthImgF   = np.delete(rigthImg, removeList, axis=0)
steeringF   = np.delete(steering, removeList, axis=0)


# 2.5.4.3 PLOT THE DISTRIBUTION OF TRAINING DATA AFTER FILTERING REDUNDANT INFORMATION.
plt.figure()
plt.hist(steeringF, bins=20)
plt.xlabel('Steering values')
plt.ylabel('Samples')
plt.title('Training set with less unbalanced data')


plt.figure()
plt.hist(steering, bins=20)
plt.hist(steeringF, bins=20)
hist2, bins2 = np.histogram(steeringF, bins=nbins)
plt.xlabel('Steering values')
plt.ylabel('Samples')
plt.title('Training set distribution before and after filtering data')



# 3. (Preprocessing part 3) IMAGE PREPROCESSING BEFORE FEEDING THE MODEL

# 3.1 Function to read image
def read_image(path, file_name):
    '''
    load image in RGB color space
    '''
    import matplotlib.image as mpimg
    import os
    return mpimg.imread(os.path.join(path, file_name))

plt.figure()
plt.title('Center camera Image example in RGB.')
plt.imshow(read_image(path_in, centerImg[1]))


# 3.2 Function to crop image
def crop_region(img):
    '''
    crops to desired region
    img: input image
    '''
    cropped = img[60:-20, :, :]
    return cropped


plt.figure()
plt.title('Cropped image in RGB.')
plt.imshow(crop_region(read_image(path_in, centerImg[1])))

# 3.3 Function to resize image
def resize_img(img, output_shape=(64,64)):
    '''
    returns resized image to the given "output_shape"
    output_shape: Desired output shape
    '''
    import cv2
    
    shape= output_shape
    resized = cv2.resize(img, shape)
    return resized

plt.figure()
plt.title('Resized image in RGB.')
plt.imshow(resize_img(crop_region(read_image(path_in, centerImg[1]))))

# 3.4 Function to convert RGB in YUV color space
def rgb2yuv(img):
    '''
    returns image in YUV color space.
    '''
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


plt.figure()
plt.title('Center image in YUV.')
plt.imshow(rgb2yuv(resize_img(crop_region(read_image(path_in, centerImg[1])))))



# 3.5 Select randomly an inage from center. left or rigth camera.
def random_image_select(path, centerImgName, leftImgName, rigthImgName, angle):
    '''
    select randomly among center, left and rigth images and computes their respective angle correction/adjustment.
    '''
    import numpy as np
    
    selector = np.random.randint(3)
    
    if selector == 0:
        return read_image(path, centerImgName) , angle
    elif selector == 1:
        return read_image(path, leftImgName) , angle + 0.2
    else:
        return read_image(path, rigthImgName) , angle - 0.2



# 3.6 Select randomly iamges to flip horizontally,.
def random_flip(img, angle):
    '''
    Pseudo random flips input image horizontally.
    '''
    import numpy as np
    import cv2
    
    if np.random.rand() > 0.5:
        mirror_img = cv2.flip(img, 1)
        mirror_angle = -angle
        return mirror_img, mirror_angle
    else:  
        return img, angle
    
a = read_image(path_in, centerImg[1])
plt.imshow(random_flip(a, steeringF[1])[0])
plt.title('Center camera image flipped example.')

# 3.7 . Rnadomly apply shadow to images.
def random_shadow(img):
    '''
    Returns image section with shadow
    '''
    import cv2
    import numpy as np
    
    # y_max = image_high, x_max= image_width, depth = image_channels
    y_max, x_max, depth = np.shape(img)
    
    # A linear region from original image will be selected for random shadow addition.
    # Note: For the equations: y_grid and x_grid are variable. others (x1,y1,x2,y2) are known values
    # p1 = (x1,y1) , p2 = (x2, y2)    , px = (x_grid, y_grid)
    
    # m1 > m2 , that is (region above line, will be filled with ones)
    # (y_grid - y1) / (x_grid - x1) > (y2 -y1) / (x2 - x1)
    # That is equivalent to: (ygrid - y1)*(x2 -x1) - (y2-y1)*(xgrid- x1) > 0
    x1 = int(x_max* np.random.rand()) # known value
    x2 = int(x_max* np.random.rand()) # known value
    y1 = 0      # known value
    y2 = y_max  # known value

    xgrid, ygrid = np.mgrid[0:y_max, 0:x_max]   # variable values
    mask = np.zeros_like(img[:,:,1])
    mask[(ygrid - y1)*(x2 -x1) - (y2-y1)*(xgrid- x1) > 0 ] = 1
    
    # select if shadow will be applied to upper or lower region
    selected = (mask == np.random.randint(2))
    saturation_ratio = np.random.uniform(low=0.3, high= 0.7)
    
    # adjust Saturation in HLS color space(Hue, Ligth, Saturation)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls[:,:,1][selected] = hls[:,:,1][selected] * saturation_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

# plot 5 images from training set after appliying random shaddow.
for i in range (5):
    a = read_image(path_in, centerImg[i])
    plt.figure()
    plt.imshow(random_shadow(a))
    plt.title('Random shadow example.')


# 3.9 Brigth augmentation
def bright_augmentation2(img, alpha= 0.5, beta= 0.0):
    '''
    returns image with 0.1 random uniform brigthness noise addition
    img: image to add random bright noise
    '''
    import numpy as np
    import cv2
    
    k = 1.0 + alpha*np.random.uniform(low = -0.5, high= 0.5) + beta
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img2[:,:,2] = img2[:,:,2] * k
    new_img = cv2.cvtColor(img2, cv2.COLOR_HSV2RGB)
    
    return new_img

for i in range (5):
    a = read_image(path_in, centerImg[i*100])
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(a)
    plt.title('Image example.')
    plt.subplot(1,2,2)
    plt.imshow(bright_augmentation2(a))
    plt.title('Brightness augmentation example.')



# 4. (Preprocessing part 4) PRERPOCESS TRAINING DATA PIPELINES

# 4.1. basic data preprocessing
def preprocess_img(img):
    '''
    Returns preprocessed image
    '''
    img2 = crop_region(img)
    img2 = resize_img(img2)
    
    return img2

# 4.2 More detain training data preprocessing
def preprocess_train_data(path, centerImgName, leftImgName, rigthImgName, steeringAngle):
    '''
    Applies predefined functions to training data
    '''
    
    img, angle = random_image_select(path, centerImgName, leftImgName, rigthImgName, steeringAngle)
    
    img, angle = random_flip(img, angle)
    img = random_shadow (img)
    img = bright_augmentation2(img)
    
    return img, angle


# 4.3 Example of an image after training preprocess.
ax, ay = preprocess_train_data('./sim_data_15/IMG/', centerImgF[0], leftImgF[0], rigthImgF[0], steeringF[0])
plt.imshow(ax)
plt.title('Preprocessed training image example.')

del ax, ay


# 5. BATCH GENERATOR DEFINITION

# 5.1 This batch generator is used just for debugging. It is not the one used for the trainign an validation sets.
def batch_generator2(path_dir, path_centers, path_lefts, path_rigths, steering_angles, img_shape, batch_size=32, flg_train=0):
    '''
    Generates training batch (images and corresponding steering_angles)
    '''
    from sklearn.utils import shuffle
    
    imgHeigh, imgWidth, imgDepth = img_shape
    
    # create images and angles tensors
    imgs = np.zeros([batch_size, imgHeigh, imgWidth, imgDepth], dtype=np.uint8) # for images dtype= np.uint8
    angs = np.zeros(batch_size) # dtype= np.float64 for angles
    
    # preprocess data 
    cImg, lImg, rImg, sAngle = shuffle(path_centers, path_lefts, path_rigths, steering_angles)
    for i in range(batch_size):
        
        
        centerImg, leftImg, rigthImg, angle = cImg[i], lImg[i], rImg[i], sAngle[i]
        
        if (flg_train == 1) and (np.random.rand() > 0.5):
            imgData , angle = preprocess_train_data(path_dir, centerImg, leftImg, rigthImg, angle)
        else:
            imgData  = read_image(path_dir, centerImg)
            
        # add image and steering angle to batch
        imgs[i], angs[i] = preprocess_img(imgData), angle
    
    return imgs, angs
            

# 5.2. Using: " batch_generator2()" funtion for debugging.
images, angles = batch_generator2(path_dir=path_dir, path_centers=centerImgF, path_lefts=leftImgF, path_rigths=rigthImgF, steering_angles=steeringF, img_shape=[64, 64, 3], batch_size=64, flg_train=1)

nx = 1
plt.imshow(images[nx])
print(angles[nx])
angles
plt.hist(steeringF)
del images, angles


# 5.3. BATCH GENERATOR FOR TRAINING AND VALIDATION
def batch_generator(path_dir, path_centers, path_lefts, path_rigths, steering_angles, img_shape, batch_size=32, flg_train=0):
    '''
    Generates training batch (images and corresponding steering_angles)
    '''
    from sklearn.utils import shuffle
    
    imgHeigh, imgWidth, imgDepth = img_shape
    
    # create images and angles tensors
    imgs = np.zeros([batch_size, imgHeigh, imgWidth, imgDepth], dtype=np.uint8) # for images dtype= np.uint8
    angs = np.zeros(batch_size) # dtype= np.float64 for angles
    
    while True: # loop forever
        # preprocess data 
        
        for i in range(batch_size):
            cImg, lImg, rImg, sAngle = shuffle(path_centers, path_lefts, path_rigths, steering_angles)    
            
            centerImg, leftImg, rigthImg, angle = cImg[i], lImg[i], rImg[i], sAngle[i]
            
            if (flg_train == 1) and (np.random.rand() > 0.5):
                imgData , angle = preprocess_train_data(path_dir, centerImg, leftImg, rigthImg, angle)
            else:
                imgData  = read_image(path_dir, centerImg)
                
            # add image and steering angle to batch
            imgs[i], angs[i] = preprocess_img(imgData), angle
        
        yield imgs, angs



# 5.4 GENERATE BATCH_TrAINING and BATCH_VALIDATION
#  img_shape=[64, 64, 3]
#  batch_size=64
batch_train = batch_generator(path_dir=path_dir, path_centers=centerImgF, path_lefts=leftImgF, path_rigths=rigthImgF, steering_angles=steeringF, img_shape=[64, 64, 3], batch_size=64, flg_train=1)
batch_valid = batch_generator(path_dir=path_dir, path_centers=X_valid, path_lefts=leftImgV, path_rigths=rigthImgV, steering_angles=y_valid, img_shape=[64, 64, 3], batch_size=64, flg_train=0)




# 6.. DEFINE MODEL AND TRAIN IT
# model was defined using keras on top.

# 6.1 Define input image shape

row, col, ch = 64, 64, 3

# 6.1 DEFINE MODEL
model = Sequential()

# Input planes 3 @ 75x120
# For normalization I add a Lambdayer to the model
# Nomalized Input planes 3 @ 75x120
model.add(Lambda(lambda x:x /255.0 - 0.5, input_shape=(row,col,ch)))

# Layer 1: Convolutional feature map
model.add(Convolution2D(24, kernel_size=(5,5), strides=(2,2), padding='valid'))
model.add(Activation('elu'))

# Layer2: Convolutional feature map
model.add(Convolution2D(36, kernel_size=(5,5), strides=(2,2), padding='valid'))
model.add(Activation('elu'))

# Layer 3: Convolutional feature map
model.add(Convolution2D(48, kernel_size=(5,5), strides=(2,2), padding='valid'))
model.add(Activation('elu'))


# Layer 4: Convolutional feature map
model.add(Convolution2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('elu'))


# Layer 5: Convolutional feature map
model.add(Convolution2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('elu'))

# Layer 6: DROP OUT layer
model.add(Dropout(0.5))

# Flatten Layer: - Add a flatten layer
model.add(Flatten())

# Layer 7: First Fully connected layer
model.add(Dense(100, W_regularizer = l2(0.001)))
# Layer 8: Dropout layer
model.add(Dropout(0.5))
# Layer 9: Seccond fully connected layer.
model.add(Dense(50, kernel_regularizer = l2(0.001)))
# Layer 10: Dropout layer.
model.add(Dropout(0.5))
# Layer 11: Fully connected layer. output layer..
model.add(Dense(1, kernel_regularizer= l2(0.001)))



#. 6.2 Compile model and generate model summary.
model.compile(loss='mse', optimizer='adam')
model.summary()


#. 6.3. Traing model and check training and validation accuracy reported.
batch_size = 64
fit_model = model.fit_generator(generator=batch_train, steps_per_epoch=int(len(centerImgF)/batch_size), epochs=30, verbose=1, validation_data=batch_valid, validation_steps=int(len(centerImgV)/batch_size), max_queue_size=1, workers=4)#, use_multiprocessing=True)

# 6.5. save model to test it in simulator.
model.save('model.h5')









