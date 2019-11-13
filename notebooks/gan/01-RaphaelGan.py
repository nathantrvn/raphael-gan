#!/usr/bin/env python
# coding: utf-8

# # Genesis of Raphaël
# 
# Unleashing the power of DCNN to create random Raphaëls.

# In[26]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Dropout, Input, Dense, Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model

SAVE_PATH = "../../models"


# In[68]:


def build_generator(latent_size):
    
    generator = Sequential([
        # (latent_size)
        Input(shape=(latent_size,), name="LatentNoise"),
        Dense(16*16*128, activation="relu"),
        # (16, 16, 384)
        Reshape((16, 16, 128)),
        # (32, 32, 192)
        UpSampling2D(),
        Conv2DTranspose(64, 3, padding="same", activation="relu"),
        BatchNormalization(momentum=0.8),
        # (64, 64, 96)
        UpSampling2D(),
        Conv2DTranspose(32, 3, padding="same", activation="relu"),
        BatchNormalization(momentum=0.8),
        # (128, 128, 3)
        UpSampling2D(),
        Conv2DTranspose(3, 3, padding="same", activation="tanh")
    ], name="generator")
    
    return generator

def build_discriminator():
    
    discriminator = Sequential([
        # (128, 128, 3)
        Input(shape=(128, 128, 3), name="InputImages"),
        Conv2D(32, 3, padding="same", kernel_initializer="he_normal"),
        LeakyReLU(0.2),
        Conv2D(32, 3, padding="same", kernel_initializer="he_normal"),
        LeakyReLU(0.2),
        MaxPooling2D(),
        Dropout(0.2),
        
        Conv2D(64, 3, padding="same", kernel_initializer="he_normal"),
        LeakyReLU(0.2),
        Conv2D(64, 3, padding="same", kernel_initializer="he_normal"),
        LeakyReLU(0.2),
        MaxPooling2D(),
        Dropout(0.2),
        
        Conv2D(128, 3, padding="same", kernel_initializer="he_normal"),
        LeakyReLU(0.2),
        Conv2D(128, 3, padding="same", kernel_initializer="he_normal"),
        LeakyReLU(0.2),
        MaxPooling2D(),
        
        Flatten(),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(2, activation="softmax")        
    ], name="discriminator")
    
    return discriminator


# In[69]:


generator = build_generator(100)


# In[70]:


discriminator = build_discriminator()


# In[71]:


discriminator.summary()


# In[72]:


generator.summary()


# In[79]:


def build_combined(discriminator, generator, latent_size):
    noise = Input(shape=(latent_size, ), name="noise")
    fake_raph = generator(noise)
    discriminator.trainable = False
    validation = discriminator(fake_raph)
    return Model(noise, validation, name="combined")


# In[80]:


combined = build_combined(discriminator, generator, 100)


# In[81]:


combined.summary()


# In[ ]:




