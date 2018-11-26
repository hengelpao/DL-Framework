import cv2
import os
import numpy as np
from dataset import prepareData
from frontend import DLFramework

def runTrain(network_name):
    
    # Training parameter setup
    epochs              = 20
    batch_size          = 48
    lr_rate             = 1e-4
    image_w             = 320
    image_h             = 320
    weight_path         = network_name + '_model.h5'

    # Training data folder
    input_folder        = 'data/input/'
    output_folder       = 'data/output/'

    train_images = prepareData(input_folder,output_folder)
    np.random.shuffle(train_images)

    # We split the training data into 80% train data and 20% validation data
    split = int(0.8*len(train_images))
    valid_images = train_images[split:]
    train_images = train_images[:split]

    # We call the network class
    object = DLFramework(network_name        = network_name,
                            image_h          = image_h,
                            image_w          = image_w)

    # If it exists, we call the pretrained weights
    if(os.path.exists(weight_path)):
        object.load_weights(weight_path)

    # We call train function to train
    object.train(train_images   = train_images,
                 valid_images   = valid_images,
                 batch_size     = batch_size,
                 lr_rate        = lr_rate,
                 weight_path    = weight_path,
                 epochs         = epochs)



def runPredict(network_name):
    # Predict parameter setup
    image_w             = 320
    image_h             = 320
    weight_path         = network_name + '_model.h5'

    # We call the network class
    object = DLFramework(network_name        = network_name,
                            image_h          = image_h,
                            image_w          = image_w)

    # If it exists, we call the pretrained weights
    if(os.path.exists(weight_path)):
        object.load_weights(weight_path)

    # We call predict function to predict
    object.predict('general_data')

def runEvaluation(network_name):
    # Predict parameter setup
    image_w             = 320
    image_h             = 320
    weight_path         = network_name + '_model.h5'

    # We call the network class
    object = DLFramework(network_name        = network_name,
                            image_h          = image_h,
                            image_w          = image_w)

    # If it exists, we call the pretrained weights
    if(os.path.exists(weight_path)):
        object.load_weights(weight_path)

    # We call predict function to predict
    object.evaluate('general_data')


if __name__ == '__main__':
    runTrain('NEW')
    # runPredict('NEW')
    # runEvaluation('NEW')