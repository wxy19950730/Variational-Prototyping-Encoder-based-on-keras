# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 09:55:30 2020

@author: HASEE
"""
import numpy as np
import os
import random
import skimage.io as io
import skimage.transform as transform


from keras.layers import Dense,Activation,Lambda,Input,Flatten,Reshape,MaxPooling2D,Concatenate,UpSampling2D
from keras.layers.convolutional import Conv2D
import keras.backend as K
from keras.models import Model,load_model
from keras import objectives
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization


#定义模型相关参数
#batch_size=32
shape_1=(64,64,3)
#intermediate_dim=64
latent_dim=300

#定义encoder
x = Input(shape=shape_1)

h1 = Conv2D(filters=100,kernel_size=(3,3),padding='same',strides=2)(x)
h1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                        beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                        moving_variance_initializer='ones')(h1)
h1 = LeakyReLU(alpha=0.3)(h1)
h2 = Conv2D(filters=150,kernel_size=(4,4),padding='same',strides=2)(h1)
h2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                        beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                        moving_variance_initializer='ones')(h2)
h2 = LeakyReLU(alpha=0.3)(h2)
h3 = Conv2D(filters=250,kernel_size=(4,4),padding='same',strides=2)(h2)
h3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                        beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                        moving_variance_initializer='ones')(h3)
h3 = LeakyReLU(alpha=0.3)(h3)

h4 = Flatten()(h3)
z_mean = Dense(latent_dim)(h4)
z_log_sigma = Dense(latent_dim)(h4)

def sampling(args):#重参数化技巧
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_mean),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,),name='encoder_output')([z_mean, z_log_sigma])
encoder = Model(x, z)

#定义解码器
d1 = Dense(16000, activation='relu')(z)
d1 = Reshape((8,8,250))(d1)
d1 = UpSampling2D(size=(2, 2),dim_ordering='tf')(d1)
d2 = Conv2D(filters=150,kernel_size=(3,3),padding='same',strides=1)(d1)
d2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                        beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                        moving_variance_initializer='ones')(d2)
d2 = LeakyReLU(alpha=0.3)(d2)
d2 = UpSampling2D(size=(2, 2),dim_ordering='tf')(d2)
d3 = Conv2D(filters=100,kernel_size=(3,3),padding='same',strides=1)(d2)
d3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                        beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                        moving_variance_initializer='ones')(d3)
d3 = LeakyReLU(alpha=0.3)(d3)
d3 = UpSampling2D(size=(2, 2),dim_ordering='tf')(d3)
d4 = Conv2D(filters=3,kernel_size=(3,3),padding='same',strides=1)(d3)
d4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                        beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                        moving_variance_initializer='ones')(d4)

vpe = Model(x, d4)
#vpe.summary()

def vae_loss(y_true, y_pre):
    xent_loss = objectives.binary_crossentropy(y_true, y_pre)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vpe.compile(optimizer='rmsprop', loss=vae_loss)

#step_per_epoch = 0
def generate_dataset():
    
    while(1):
    
        true_image_dir = r'.\vpe_dataset\train\true_image'
        prototyping_image_dir = r'.\vpe_dataset\train\Prototyping_image'
        
        class_list = os.listdir(true_image_dir)
        
        true_files_list=[]
        prot_files_list=[]
        
        for path in class_list:
            
            true_dir_path = ''.join([true_image_dir,"\\",path])
            prot_file_path = ''.join([prototyping_image_dir,"\\",path,".jpg"])
            
            file_list = os.listdir(true_dir_path)
            for file in file_list:
                file_path = ''.join([true_dir_path,"\\",file])
                true_files_list.append(file_path)
                prot_files_list.append(prot_file_path)
                
        if len(true_files_list) != len(prot_files_list):
            raise Exception("原型图像数目与真实图像种类数不匹配", [len(prot_files_list),len(true_files_list)])
            
        global step_per_epoch
        step_per_epoch = len(true_files_list)
        
        index_list = [x for x in range(len(true_files_list))]
        random.shuffle(index_list)
        for i in index_list:
            x = (io.imread(true_files_list[i])).astype("float32")/255.
            x = np.reshape(transform.resize(x, shape_1),(1,shape_1[0],shape_1[1],shape_1[2]))
            y = (io.imread(prot_files_list[i])).astype("float32")/255.
            y = np.reshape(transform.resize(y, shape_1),(1,shape_1[0],shape_1[1],shape_1[2]))
            #yield ({'input': x}, {'output': y})
            yield (x,y)
            
history = vpe.fit_generator(generate_dataset(),samples_per_epoch=875,nb_epoch=30)        

    


test_image = io.imread(r'E:\buaa-project\Few-Shot_Learning_with_Variational_Semantic_Autoencoder_Based_on_Engineering_Model\test_codes\vpe_dataset\test\true_image\no_entry\\02396_00000.png').astype("float32")/255.
test_image = np.reshape(transform.resize(test_image, (64,64,3)),(1,64,64,3))
output = np.reshape(vpe.predict(test_image),(64,64,3))
io.imshow(output)
vpe.save(r'vpe_20epoch_20200824.h5')
encoder.save(r'encoder_20epoch_20200824.h5')



#model = load_model(r'20_epoch.h5', custom_objects={'vae_loss': vae_loss} )

