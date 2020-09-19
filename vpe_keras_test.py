# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:39:53 2020

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

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
#encoder = Model(x, z)

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



#vpe = load_model(r'20_epoch.h5', custom_objects={'vae_loss': vae_loss})
encoder = load_model(r'encoder_20epoch_20200824.h5')


import matplotlib.pyplot as plt
from sklearn import manifold

#intermed_tensor_func=K.function([model.layers[0].input],[model.layers[-1].output])
#intermed_tensor=intermed_tensor_func([Test_data])[0]

test_true_image_dir = r'.\vpe_dataset\test\true_image' 
class_list = os.listdir(test_true_image_dir)

true_files_list=[]
file_class_list=[]
path_class = 1
for path in class_list:
    
    true_dir_path = ''.join([test_true_image_dir,"\\",path])
    file_list = os.listdir(true_dir_path)
    
    for file in file_list:
        file_path = ''.join([true_dir_path,"\\",file])
        true_files_list.append(file_path)
        file_class_list.append(path_class)
        
    path_class = path_class+1
        
#random.shuffle(true_files_list)
test_dataset = true_files_list
test_class = file_class_list
x_list = []
for i in test_dataset:
        x = (io.imread(i)).astype("float32")/255.
        x = np.reshape(transform.resize(x, shape_1),(1,shape_1[0],shape_1[1],shape_1[2]))
        x = np.reshape(encoder.predict(x),(1,300))
        x_list.append(x)
        
all_x = np.concatenate(x_list,axis=0)
#all_output = encoder.predict(all_x)

tsne=manifold.TSNE(n_components=2,init='pca',random_state=1000,perplexity=30,)
intermed_tsne=tsne.fit_transform(all_x)
#intermed_tsne = np.reshape(intermed_tsne,[200,-1])
#print('Origin data dimension is {}.Embedded data dimension is {}'
#.format(all_x[0,:].shape(-1),intermed_tsne.shape[-1]))

plt.figure(figsize=(12,12))
for i in range(len(all_x)):
	plt.scatter(intermed_tsne[i,0],intermed_tsne[i,1],c=plt.cm.Set1(test_class[i]))
    #plt.scatter(intermed_tsne[i,0],intermed_tsne[i,1])


support_file_path =r".\vpe_dataset\support_dataset"
prot_list = []
class_list = []
for path in os.listdir(support_file_path):
    prot_dir_path = ''.join([support_file_path,"\\",path])
    prot_list.append(prot_dir_path)
    class_list.append((path.split('.'))[0])
    
lantent_list = []
for i in prot_list:
        x = (io.imread(i)).astype("float32")/255.
        x = np.reshape(transform.resize(x, shape_1),(1,shape_1[0],shape_1[1],shape_1[2]))
        x = np.reshape(encoder.predict(x),(1,-1))
        lantent_list.append(x)
        
test_image_path = r".\vpe_dataset\test\true_image\stop\00148_00001.png"
x = (io.imread(test_image_path)).astype("float32")/255.
x = np.reshape(transform.resize(x, shape_1),(1,shape_1[0],shape_1[1],shape_1[2]))
test_lantent = np.reshape(encoder.predict(x),(1,-1))

lantent_list.append(test_lantent)
from sklearn.neighbors import NearestNeighbors
inquire_dataset = np.concatenate(lantent_list,axis=0)
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(inquire_dataset)
distances, indices = nbrs.kneighbors(inquire_dataset)

















#all_x = np.concatenate(x_list,axis=0)
    
























