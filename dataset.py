import cv2
import os
import numpy as np
# We refer to https://keras.io/utils/#sequence for Keras sequence
from keras.utils import Sequence
# We refer to https://github.com/aleju/imgaug for data augmentation
import imgaug as ia
from imgaug import augmenters as iaa

# This function is to prepare the path of the training data
def prepareData(input_folder, output_folder):
    train_images = []

    for filename in sorted(os.listdir(input_folder)):
        # filename is the filename of each data in the input_folder
        image = {}

        # With the assumption that the input and output have similar name:
        image['input'] = input_folder + filename
        image['output'] = output_folder + filename
        
        # Each image filename is added to train_images array
        train_images += [image]
                        
    return train_images

# This class is the dataset generator for fit_generator function in keras
class DatasetGenerator(Sequence):
    def __init__(self, images, 
                       config,
                       shuffle  =   True,
                       train    =   True,
                       norm     =   None):

        self.images         = images    # This is the array of training images
        self.shuffle        = shuffle   # It shuffles the train images if True
        self.train          = train     # It augment the train images if true
        self.norm           = norm      # It performs the normalization for each train image
        self.batch_size     = config['batch_size']    
        self.image_w        = config['image_w']
        self.image_h        = config['image_h']
        self.channel        = config['channel']
        
        self.seq            = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(scale=(0.5, 0.7))])

        # Perform the shuffling for the train images
        if self.shuffle:
            np.random.shuffle(self.images)

    # Every sequence must implement __len__ and __getitem__ functions
    # __len__ returns the number of iteration for each epoch
    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.batch_size))

    # __getitem__ returns the data for each batch_size
    def __getitem__(self, idx):
        
        # The left and right boundary index in the array for each batch_size
        l_idx = idx     * self.batch_size
        r_idx = (idx+1) * self.batch_size

        
        # If the right boundary index is larger than the total array, then it shift the array a little bit
        if r_idx > len(self.images):
            r_idx = len(self.images)
            l_idx = r_idx - self.batch_size
            
        # The array for each batch_size
        input_batch     = np.zeros((self.batch_size, self.image_h, self.image_w, self.channel))                          # input images
        output_batch    = np.zeros((self.batch_size, self.image_h, self.image_w, self.channel))  
        
        index = 0
        for image in self.images[l_idx:r_idx]:            
            input, output = self.loadData(image, train=self.train)
            
            # Assign input and output to batch array and perform normalization if needed
            if self.norm != None: 
                input_batch[index] = self.norm(input)
                output_batch[index] = self.norm(output)
            else:
                input_batch[index] = input
                output_batch[index] = output

            # Increase the index in batch array
            index += 1

        return input_batch, output_batch

    # Every sequence might have on_epoch_end function to perform any operation after each epoch ends
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

    # Function loadData performs loading operation for each input and output image
    def loadData(self, image, train):

        # Load the image using OpenCV function
        input_name = image['input']
        input = cv2.imread(input_name)
        output_name = image['output']
        output = cv2.imread(output_name)
                
        # Perform data augmentation for training process
        if train:
            seq_det = self.seq.to_deterministic()
            input = seq_det.augment_images(input)
            output = seq_det.augment_images(output)
            
        # Resize the image input desired image size for training and validation
        input = cv2.resize(input, (self.image_w, self.image_h))
        output = cv2.resize(output, (self.image_w, self.image_h))

        return input, output
