# Import the modules to read the train data.

import csv
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
from keras.backend import image_data_format


# Define data augmentation pipeline
#
# This pipeline is inspired from the post by one of the past students described
# here: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def randomBrightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    bright = np.random.uniform() + 0.25
    vChannel = img[:, :, 2]
    vChannel *= bright
    vChannel[vChannel > 255] = 255
    img[:, :, 2] = vChannel
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def randomTranslate(img, angle, nPixelsX, nPixelsY):
    nR, nC = img.shape[:-1]
    tx = nPixelsX * np.random.uniform() - nPixelsX/2
    angle += (tx/nPixelsX)*2*0.25
    ty = nPixelsY * np.random.uniform() - nPixelsY/2
    # translation matrix
    m = np.array([[1, 0, tx],
                  [0, 1, ty]], dtype=np.float32)
    img = cv2.warpAffine(img, m, (nC, nR))
    return img, angle

def randomShadow(img):
    xMax = 0
    xMin = 160
    
    yMax = 320*np.random.uniform()
    yMin = 320*np.random.uniform()
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    shadow_mask = 0*img[:,:,1]
    X_m = np.mgrid[0:img.shape[0],0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0],0:img.shape[1]][1]
    shadow_mask[((X_m - xMax) * (yMin - yMax) - (xMin - xMax) * (Y_m - yMax) >= 0)] = 1

    if np.random.randint(2)==1:
        random_bright = .25
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            img[:,:,1][cond1] = img[:,:,1][cond1]*random_bright
        else:
            img[:,:,1][cond0] = img[:,:,1][cond0]*random_bright    
    img = cv2.cvtColor(img,cv2.COLOR_HLS2RGB)
    return img

def loadRandomImg(dataRow, trainFlag=True):
    idx = np.random.randint(3)
    if idx == 0:
        imgPath = dataRow['center'].values[0]
        angleCorr = 0
    if idx == 1:
        imgPath = dataRow['left'].values[0]
        angleCorr = 0.25
    if idx == 2:
        imgPath = dataRow['right'].values[0]
        angleCorr = -0.25
        
    angle = data['steering'].values[0] + angleCorr
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    if trainFlag:
#         img = randomShadow(img)
        img, angle = randomTranslate(img, angle, 100, 50)
        img = randomBrightness(img)
        flip = np.random.randint(2)
        if flip:
            img = cv2.flip(img, 1)
            angle = -angle
    
    return img, angle


# generator for feeding into the keras model during training
def generateData(data, straightAngleRatio=0.3, batchSize=128, trainFlag=True):
    
    batchImgs = np.zeros((batchSize, 160, 320, 3), dtype=np.float32)
    batchAngles = np.zeros((batchSize,), dtype=np.float32)
    while True:
        straightCount = 0
        shuffle(data)
        for i in range(batchSize):
            idx = np.random.randint(len(data))
            rowData = data.iloc[[idx]]
            rowAngle = rowData['steering']
            
            # Limit angles of less than absolute value of 0.2 to no more than 40% of data
            # to reduce bias of car driving straight
            if abs(rowAngle.values[0]) < 0.2:
                straightCount += 1
            if straightCount > batchSize * straightAngleRatio:
                while abs(data.iloc[[idx]]['steering'].values[0]) < 0.2:
                    idx = np.random.randint(len(data))
                rowData = data.iloc[[idx]]
            
            image, angle = loadRandomImg(rowData, trainFlag)
            batchImgs[i] = image
            batchAngles[i] = angle
            
        yield batchImgs, batchAngles


# Define the model for training
# 
# The model used for this training is inspired from the model published by CommaAI 
# which can be found here: https://github.com/commaai/research/blob/master/train_steering_model.py). 
# However, I am cropping the image to get rid of the top and bottom of image as opposed to their model 
# which takes in the entire image.


def getModel():
    if image_data_format() == 'channels_first':
        inputShape = (3, 160, 320)
    else:
        inputShape = (160, 320, 3)
        
    model = Sequential()
    model.add(Lambda(lambda x: x/225.0 - 0.5, input_shape=inputShape, output_shape=inputShape))
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=inputShape))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='same'))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(ELU())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.2)) 
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))
    
    return model


# Train the model for 10 epochs and save the model. If fine turning needed, 
# load the saved model and then train with a lower learning rate. 
# Starting learn rate is set to 1e-3.


# if __name__=="__main__":

firstRun = False        # This can be input by the parser
lastEpoch = 5           # this can be input by the parser
moreEpochs = 5          # this can be input by the parser
newEpoch = moreEpochs + lastEpoch
if firstRun:
    # learning hyperparameters
    learnRate = 0.001
    straightRatio = 0.5

    # create the model
    model = getModel()

    # Load the data to train the model
    data = pd.read_csv('data/driving_log_udacity.csv')

    # append the data path with appropriate folder name
    data['center'] = data['center'].apply(lambda x: 'data/' + x)
    data['left'] = data['left'].apply(lambda x: 'data/' + x)
    data['right'] = data['right'].apply(lambda x: 'data/' + x)

else:
    learnRate = 0.0009
    straightRatio = 0.5

    # load the last saved model and parameters
    model = load_model('model.h5')

    # Load the data to train the model
    data = pd.read_csv('data/driving_log_udacity.csv')

    # append the fine tune data
    dataFineTune = pd.read_csv('data/driving_log_fine_tune.csv')
    data.append(dataFineTune)

    # append the data path with appropriate folder name
    data['center'] = data['center'].apply(lambda x: 'data/' + x)
    data['left'] = data['left'].apply(lambda x: 'data/' + x)
    data['right'] = data['right'].apply(lambda x: 'data/' + x)

plotHist = False

if plotHist:
    # the distribution in the provided dataset
    angles = np.concatenate((data['steering'].values, data['steering'].values, data['steering'].values))
    num_bins = 30
    hist, bins = np.histogram(angles, num_bins)
    width = 0.9 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.savefig('imbalanced.jpg')

    # the distribution in the dataset generated after augmentation
    generator = generateData(data)
    newAngles = np.array([], dtype=np.float32)

    for i in range(200):
        X, y = next(generator)
        newAngles = np.concatenate((newAngles, y))

    hist, bins = np.histogram(newAngles, num_bins)
    width = 0.9 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.savefig('balanced.jpg')

# train the model on the 
trainDataGenerator = generateData(data, straightAngleRatio=straightRatio)
validationDataGenerator = generateData(data, straightAngleRatio=straightRatio, trainFlag=False)

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

optim = Adam(lr=learnRate, decay=0.96)
model.compile(optimizer=optim, loss='mse')

history_object = model.fit_generator(trainDataGenerator, steps_per_epoch=200, validation_data=validationDataGenerator,
                                     validation_steps=200*0.2, epochs=newEpoch, verbose=1, 
                                     callbacks=[checkpoint], initial_epoch=lastEpoch)


# save the model checkpoints
model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

# plot the loss history
# 
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.savefig('loss.jpg')
