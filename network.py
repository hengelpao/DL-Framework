# You can call the needed layers only
# Please call the packages that are required to define the network architecture
from keras.models import Model
import tensorflow as tf
from keras.layers import * 
from keras import backend as K

# This is the master class of the network, in case you build may build many networks
class DLNetwork(object):
    # This is the abstract init function
    def __init__(self, image_w, image_h, channel):
        raise NotImplementedError('You have to implement it in the derived class!')

    # This is the normalize function for all class
    def normImage(self, image):
        return (image / 127.5) - 1


# The newNetwork class is derived from the master class 'DLNetwork', it will share the normImage function as its function too
class newNetwork(DLNetwork):
    # This is the init function for newNetwork class
    def __init__(self, image_w, image_h, channel):

        # This is the input in the network
        input = Input(shape=(image_h,image_w,3))

        # Implement your network architecture here!
        x = Conv2D(3,kernel_size=3,padding='same',kernel_initializer='he_normal')(input)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(3,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(3,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(3,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(3,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # This is the output in the network
        output = Conv2D(3,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)

        # The model is generated based on the input and output, it is saved in self.model
        self.model = Model(input,output)