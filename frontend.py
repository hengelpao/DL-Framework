import cv2
import os
import numpy as np
import tensorflow as tf

from network import newNetwork
from dataset import DatasetGenerator

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.models import Model


class DLFramework():
    def __init__(self, network_name, image_w, image_h):

        self.network_name = network_name
        self.image_w = image_w
        self.image_h = image_h
        self.channel = 3
        
        if network_name == 'NEW':
            self.network = newNetwork(self.image_w,self.image_h,self.channel)
        else:
            raise Exception('Please implement the ' + network_name +' class first!')
        
        self.network.model.summary()

    # This function is for loading the pretrained weights
    def load_weights(self, weight_path):
        self.network.model.load_weights(weight_path)  

    # This function works for performing training using the given training data
    def train(self, train_images,
                    valid_images,
                    batch_size,
                    lr_rate,
                    weight_path,
                    epochs):

        self.batch_size = batch_size
        
        # This config variable consists of several parameters for Keras sequence
        generator_config = {
            'image_h'         : self.image_h, 
            'image_w'         : self.image_w,
            'channel'         : self.channel,
            'batch_size'      : self.batch_size,
        }    

        # This is the dataset generator for training images
        train_generator = DatasetGenerator(train_images, 
                                        generator_config, 
                                        norm=self.network.normImage,
                                        train=True)
        # This is the dataset generator for validation images
        valid_generator = DatasetGenerator(valid_images, 
                                        generator_config, 
                                        norm=self.network.normImage,
                                        train=False)   

        # This is the optimizer. You can choose Adam, SGD, etc. We refer to https://keras.io/optimizers/
        optimizer = Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        
        # This is the checkpoint callback to save the network weights for each epoch. We refer to https://keras.io/callbacks/
        checkpoint = ModelCheckpoint(weight_path, 
                                        monitor='val_loss', 
                                        verbose=1, 
                                        save_best_only=False, 
                                        mode='min', 
                                        period=1)

        # This is the tensorboard callback to visualize the network. We refer to # This is the checkpoint to save the network weights for each epoch. We refer to https://keras.io/callbacks/
        tensorboard = TensorBoard(log_dir=os.path.expanduser('logs/'), 
                                    histogram_freq=0, 
                                    write_graph=True, 
                                    write_images=False)
     
        # We compile the network model, using the MSE loss. We refer to https://keras.io/losses/
        self.network.model.compile(loss=['mean_squared_error'], optimizer=optimizer)

        # We perform the training using fit_generator function (https://keras.io/models/model/#methods)
        self.network.model.fit_generator(generator     = train_generator, 
                                            steps_per_epoch             = len(train_generator), 
                                            epochs                      = epochs, 
                                            verbose                     = 1,
                                            validation_data             = valid_generator,
                                            validation_steps            = len(valid_generator),
                                            callbacks                   = [checkpoint, tensorboard], 
                                            workers                     = 3,
                                            max_queue_size              = 8)

    def evaluate(self, dataset):
        test = 0
        # You can implement the evaluate function here

    def predict(self, dataset):
        test = 0
        # You can implement the predict function here