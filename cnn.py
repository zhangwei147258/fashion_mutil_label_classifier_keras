#! -*- coding:utf-8 -*-

from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.layers import Input, Lambda, concatenate
import tensorflow as tf
from keras import backend as K


class SimpleNet(object):
    def __init__(self, input_shape, classes, finalAct="softmax"):
        #default input_shape = (width, height, channel)
        self.input_shape = input_shape
        self.classes = classes
        self.finalAct = finalAct

        #chanDim = inputShape[2]
        chanDim = -1
        if K.image_data_format() == "channels_first":
            chanDim = 1
        self.chanDim = chanDim
		
	
    def build_model(self):
        model =  Sequential()
        # CONV => RELU => POOL
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=self.chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=self.chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=self.chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
            
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=self.chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=self.chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=self.chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=self.chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # use global average pooling instead of fc layer
        model.add(GlobalAveragePooling2D())
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(self.classes))
        model.add(Activation(self.finalAct))
        model.summary()

        return model
'''
class SmallerInceptionNet(object):
    def __init__(self, input_shape, classes, finalAct="softmax"):
        #default input_shape = (width, height, channel)
        self.input_shape = input_shape
        self.classes = classes
        self.finalAct = finalAct

        #chanDim = inputShape[2]
        chanDim = -1
        if K.image_data_format() == "channels_first":
            chanDim = 1
        self.chanDim = chanDim
		
    def inception_model(self, x, filters):
        [filers1,filers2,filers3,filers4] = filters
        #1*1
        branch1x1 = Conv2D(filters=filters1[0],kernel_size=(1,1), padding='same',strides=(1,1))(x)

        #1*1->3*3
        branch3x3 = Conv2D(filters=filters2[0],kernel_size=(1,1), padding='same',strides=(1,1))(x)
        branch3x3 = Conv2D(filters=filters2[1],kernel_size=(3,3), padding='same',strides=(1,1))(branch3x3)

        #1*1->5*5
        branch5x5 = Conv2D(filters=filters3[0],kernel_size=(1,1), padding='same',strides=(1,1))(x)
        branch5x5 = Conv2D(filters=filters3[1],kernel_size=(1,1), padding='same',strides=(1,1)(branch5x5)

        #maxpooling->conv(1*1)
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(2, 2)(x)
        branchpool = Conv2D(filters=filters4[0],kernel_size=(1,1),padding='same',strides=(1,1))(branchpool)

        x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)
        return x
	
    def build_model(self):

        inputs = Input(shape=self.input_shape)
        #(CONV => RELU) * 2 => POOL
        x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = MaxPooling2D(pool_size=(3,3), strides=(2, 2))(x)

        x = self.inception_model(x,[(64,), (64,128), (64, 128), (64,)])
        x = self.inception_model(x,[(128,), (64,128), (64, 128), (128,)])
        x = MaxPooling2D(pool_size=(3,3), strides=(2, 3))(x)

        x = self.inception_model(x,[(256,), (64,512), (64, 512), (64,)])
        x = self.inception_model(x,[(512,), (128,256), (256, 512), (128,)])

        x = AveragePooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation(finalAct)(x)

        model = Model(inputs=inputs, outputs=x)
        model.summary()
        return model
'''
class FashionNet(object):
    def __init__(self, input_shape, category_classes, color_classes, finalAct="softmax"):
        #default input_shape = (width, height, channel)
        self.input_shape = input_shape
        self.category_classes = category_classes
        self.color_classes = color_classes
        self.finalAct = finalAct

        #chanDim = inputShape[2]
        chanDim = -1
        if K.image_data_format() == "channels_first":
            chanDim = 1
        self.chanDim = chanDim
		
    def build_category_branch(self, inputs):
        # convert 3 channel(rgb) input to gray 
        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        #Conv->ReLU->BN->Pool
        x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = MaxPooling2D(pool_size=(3,3))(x)

        #(CONV => RELU) * 2 => POOL
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # (CONV => RELU) * 2 => POOL
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # (CONV => RELU) * 2 => POOL
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # use global average pooling instead of fc layer
        x = GlobalAveragePooling2D()(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Dense(self.category_classes)(x)
        x = Activation(self.finalAct, name='category_output')(x)

        return x
	
    def build_color_branch(self, inputs):
        #Conv->ReLU->BN->Pool
        x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = MaxPooling2D(pool_size=(3,3))(x)

        #Conv->ReLU->BN->Pool*2
        x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        #Conv->ReLU->BN->Pool*2
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=self.chanDim)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.color_classes)(x)
        x = Activation(self.finalAct, name='color_output')(x)
        return x 

    def build_model(self):
        input_shape = self.input_shape
        inputs = Input(shape=input_shape)
        category_branch = self.build_category_branch(inputs) 
        color_branch = self.build_color_branch(inputs) 

        model = Model(inputs=inputs, outputs=[category_branch, color_branch])
        model.summary()
        return model	

